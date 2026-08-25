//! Deferred write operations for transactional semantics.
//!
//! This module provides [`Pending`], which batches write operations during
//! query execution. This enables:
//!
//! - Read-your-writes within a query (created nodes visible to later clauses)
//! - Atomic commit/rollback of all changes
//! - Efficient bulk updates to indexes
//!
//! ## Batched Operations
//!
//! - `created_nodes`: Nodes created in this query
//! - `deleted_nodes`: Nodes marked for deletion
//! - `created_rels_by_type`: Edges created in this query, grouped by type
//! - `deleted_relationships`: Edges marked for deletion
//! - `set_*_attrs`: Property updates by entity ID
//! - `set/remove_node_labels`: Label changes
//!
//! ## Commit Flow
//!
//! ```text
//! Query execution → accumulate in Pending → apply_all() → update Graph
//! ```
//!
//! On error or ROLLBACK, the Pending is simply dropped without applying.

use std::{cell::RefCell, ops::BitOrAssign, sync::Arc};

use rustc_hash::FxHashMap;

use atomic_refcell::AtomicRefCell;
use roaring::RoaringTreemap;

use crate::graph::graph::DeletedEdge;
use crate::{
    entity_type::EntityType,
    graph::{
        constraint::{ConstraintStatus, ConstraintType},
        graph::{Graph, LabelId, NodeId, RelationshipId},
    },
    runtime::{ordermap::OrderMap, orderset::OrderSet, runtime::QueryStatistics, value::Value},
};

/// Flatten a node_id → [label_ids] map into parallel (rows, cols) arrays.
fn flatten_label_map(map: &FxHashMap<u64, Vec<u64>>) -> (Vec<u64>, Vec<u64>) {
    let total: usize = map.values().map(std::vec::Vec::len).sum();
    let mut rows = Vec::with_capacity(total);
    let mut cols = Vec::with_capacity(total);
    for (&node_id, label_ids) in map {
        for &label_id in label_ids {
            rows.push(node_id);
            cols.push(label_id);
        }
    }
    (rows, cols)
}

const INVALID_PROPERTY_MSG: &str =
    "Property values can only be of primitive types or arrays of primitive types";

fn is_valid_property(
    value: &Value,
    allow_null: bool,
) -> bool {
    match value {
        Value::Null => allow_null,
        Value::Bool(_)
        | Value::Int(_)
        | Value::Float(_)
        | Value::String(_)
        | Value::Point(_)
        | Value::VecF32(_)
        | Value::Datetime(_)
        | Value::Date(_)
        | Value::Time(_)
        | Value::Duration(_) => true,
        Value::List(items) => items.iter().all(|v| is_valid_property(v, false)),
        _ => false,
    }
}

/// Validate that a value is a valid node property type.
fn validate_node_property(value: &Value) -> Result<(), String> {
    if !is_valid_property(value, true) {
        return Err(INVALID_PROPERTY_MSG.into());
    }
    Ok(())
}

/// Validate that a value is a valid relationship property type.
fn validate_relationship_property(value: &Value) -> Result<(), String> {
    if !is_valid_property(value, true) {
        return Err(INVALID_PROPERTY_MSG.into());
    }
    Ok(())
}

/// Binary-search a sorted `(attr_id, Value)` slice.
fn lookup_sorted(
    attrs: &[(u16, Value)],
    attr_id: u16,
) -> Option<&Value> {
    attrs
        .binary_search_by_key(&attr_id, |(k, _)| *k)
        .ok()
        .map(|pos| &attrs[pos].1)
}

/// Accumulated write operations for deferred application.
///
/// All mutations during query execution are collected here and applied
/// atomically at the end. This enables transactional semantics.
pub struct Pending {
    /// Nodes created in this transaction
    created_nodes: RoaringTreemap,
    /// Relationships created, grouped by type: type_name → [(rel_id, from, to)]
    created_rels_by_type: FxHashMap<Arc<String>, Vec<(RelationshipId, NodeId, NodeId)>>,
    /// Reverse index: rel_id → type_name for O(1) existence/type lookups
    created_rel_types: FxHashMap<RelationshipId, Arc<String>>,
    /// Nodes to be deleted
    deleted_nodes: RoaringTreemap,
    /// Relationships to be deleted
    deleted_relationships: RoaringTreemap,
    /// Endpoints for deleted relationships — populated by commit(), used by build_effects_buffer().
    deleted_endpoints: Vec<DeletedEdge>,
    /// Property updates for newly created nodes (fast path: skip fjall).
    /// Values are attribute-id-resolved, sorted by id, unique.
    new_nodes_attrs: FxHashMap<u64, Vec<(u16, Value)>>,
    /// Property updates for existing nodes (full merge path)
    existing_nodes_attrs: FxHashMap<u64, Vec<(u16, Value)>>,
    /// Property updates for newly created relationships (fast path)
    new_relationships_attrs: FxHashMap<u64, Vec<(u16, Value)>>,
    /// Property updates for existing relationships (full merge path)
    existing_relationships_attrs: FxHashMap<u64, Vec<(u16, Value)>>,
    /// Labels to add: node_id → [label_ids]
    set_labels: FxHashMap<u64, Vec<u64>>,
    /// Labels to remove: node_id → [label_ids]
    remove_labels: FxHashMap<u64, Vec<u64>>,
    /// Documents to add to indexes (keyed by label id)
    index_add_docs: FxHashMap<u64, RoaringTreemap>,
    /// Documents to remove from indexes (keyed by label id)
    index_remove_docs: FxHashMap<u64, RoaringTreemap>,
    /// Edge documents to add to indexes (keyed by relationship type id)
    index_add_edge_docs: FxHashMap<u64, RoaringTreemap>,
    /// Edge documents to remove from indexes: `type_id → { edge_id → (src, dst) }`.
    /// `(src, dst)` is captured at deletion time — the edge is gone
    /// from the tensor by the time `commit_edge_index` runs so the
    /// 24-byte RediSearch key must be reconstructable from here.
    index_remove_edge_docs: FxHashMap<u64, FxHashMap<u64, (u64, u64)>>,
    /// Deferred index operations — accumulated across commit cycles,
    /// applied only after the full query succeeds so that a failed
    /// query never leaves stale entries in RediSearch.
    deferred_index_adds: FxHashMap<u64, RoaringTreemap>,
    deferred_index_removes: FxHashMap<u64, RoaringTreemap>,
    deferred_edge_index_adds: FxHashMap<u64, RoaringTreemap>,
    deferred_edge_index_removes: FxHashMap<u64, FxHashMap<u64, (u64, u64)>>,
    /// Union of every index document this query has published, accumulated across
    /// all of its `Commit`s, so a later failure can resync them against committed
    /// state (see [`Self::resync_published_indexes`]).
    ///
    /// Deliberately **not** reset by [`Self::clear`], which runs after every
    /// `Commit`: the undo has to cover the whole query, not just the last `Commit`.
    /// Per-query state — `Pending` belongs to one `Runtime`.
    published: DeferredIndexes,
    /// Schema baseline: number of labels when the current commit window started.
    schema_label_count: usize,
    /// Schema baseline: number of relationship types when the current commit window started.
    schema_rel_type_count: usize,
    /// Schema baseline: number of node attribute names when the current commit window started.
    schema_node_attr_count: usize,
    /// Schema baseline: number of relationship attribute names when the current commit window started.
    schema_rel_attr_count: usize,
}

/// One `Commit`'s index document changes, collected while applying `pending` and
/// written to RediSearch as a batch.
#[derive(Default)]
pub struct DeferredIndexes {
    node_adds: FxHashMap<u64, RoaringTreemap>,
    node_removes: FxHashMap<u64, RoaringTreemap>,
    edge_adds: FxHashMap<u64, RoaringTreemap>,
    edge_removes: FxHashMap<u64, FxHashMap<u64, (u64, u64)>>,
}

impl DeferredIndexes {
    /// Fold `other` in, for accumulating everything one query published.
    fn merge(
        &mut self,
        other: &Self,
    ) {
        for (slot, ids) in &other.node_adds {
            *self.node_adds.entry(*slot).or_default() |= ids;
        }
        for (slot, ids) in &other.node_removes {
            *self.node_removes.entry(*slot).or_default() |= ids;
        }
        for (slot, ids) in &other.edge_adds {
            *self.edge_adds.entry(*slot).or_default() |= ids;
        }
        for (slot, ids) in &other.edge_removes {
            self.edge_removes.entry(*slot).or_default().extend(ids);
        }
    }

    /// Every id in this set per slot, regardless of direction — all the undo path
    /// needs, since it re-derives each document from committed state.
    fn ids_by_slot(
        &self
    ) -> (
        FxHashMap<u64, RoaringTreemap>,
        FxHashMap<u64, RoaringTreemap>,
    ) {
        let mut nodes: FxHashMap<u64, RoaringTreemap> = FxHashMap::default();
        let mut edges: FxHashMap<u64, RoaringTreemap> = FxHashMap::default();
        for docs in [&self.node_adds, &self.node_removes] {
            for (slot, ids) in docs {
                *nodes.entry(*slot).or_default() |= ids;
            }
        }
        for (slot, ids) in &self.edge_adds {
            *edges.entry(*slot).or_default() |= ids;
        }
        for (slot, ids) in &self.edge_removes {
            edges.entry(*slot).or_default().extend(ids.keys().copied());
        }
        (nodes, edges)
    }

    /// Write the batch to the shared RediSearch index.
    ///
    /// Only valid in writer mode: the index is not MVCC, so readers must be
    /// excluded (mirrors C, which updates index docs inside the graph write lock).
    pub fn commit(
        &mut self,
        g: &AtomicRefCell<Graph>,
    ) {
        // This is the query's own private version, which nothing else can borrow, so
        // graph-then-indexer is safe here — see `Pending::resync_published_indexes`
        // for the published version, where the order has to be the other way round.
        let mut g = g.borrow_mut();
        g.commit_index(&mut self.node_adds, &mut self.node_removes);
        g.commit_edge_index(&mut self.edge_adds, &mut self.edge_removes);
    }
}

impl Default for Pending {
    fn default() -> Self {
        Self::new()
    }
}

impl Pending {
    #[must_use]
    pub fn new() -> Self {
        Self {
            created_nodes: RoaringTreemap::new(),
            created_rels_by_type: FxHashMap::default(),
            created_rel_types: FxHashMap::default(),
            deleted_nodes: RoaringTreemap::new(),
            deleted_relationships: RoaringTreemap::new(),
            deleted_endpoints: Vec::new(),
            new_nodes_attrs: FxHashMap::default(),
            existing_nodes_attrs: FxHashMap::default(),
            new_relationships_attrs: FxHashMap::default(),
            existing_relationships_attrs: FxHashMap::default(),
            set_labels: FxHashMap::default(),
            remove_labels: FxHashMap::default(),
            index_add_docs: FxHashMap::default(),
            index_remove_docs: FxHashMap::default(),
            index_add_edge_docs: FxHashMap::default(),
            index_remove_edge_docs: FxHashMap::default(),
            published: DeferredIndexes::default(),
            deferred_index_adds: FxHashMap::default(),
            deferred_index_removes: FxHashMap::default(),
            deferred_edge_index_adds: FxHashMap::default(),
            deferred_edge_index_removes: FxHashMap::default(),
            schema_label_count: 0,
            schema_rel_type_count: 0,
            schema_node_attr_count: 0,
            schema_rel_attr_count: 0,
        }
    }

    /// Record the current schema sizes so `build_effects_buffer` can emit
    /// EFFECT_ADD_SCHEMA / EFFECT_ADD_ATTRIBUTE for newly added entries.
    pub fn set_schema_baseline(
        &mut self,
        g: &AtomicRefCell<Graph>,
    ) {
        let graph = g.borrow();
        self.schema_label_count = graph.get_labels().len();
        self.schema_rel_type_count = graph.get_types().len();
        self.schema_node_attr_count = graph.get_node_attribute_names().len();
        self.schema_rel_attr_count = graph.get_relationship_attribute_names().len();
    }

    pub fn created_nodes(
        &mut self,
        ids: &[NodeId],
    ) {
        for id in ids {
            self.created_nodes.insert((*id).into());
        }
    }

    /// Set all attributes for a node. `attrs` must be attribute-id-resolved,
    /// sorted by id, and unique (last-wins dedup done by the caller).
    pub fn set_node_attributes(
        &mut self,
        id: NodeId,
        attrs: Vec<(u16, Value)>,
    ) -> Result<(), String> {
        for (_, value) in &attrs {
            validate_node_property(value)?;
        }
        // Empty attribute maps from CREATE without `{...}` props would otherwise
        // create an empty pinned cache entry per node on commit; skip them.
        if attrs.is_empty() {
            return Ok(());
        }
        debug_assert!(attrs.is_sorted_by_key(|(k, _)| *k));
        let is_new = self.created_nodes.contains(id.into());
        if is_new {
            self.new_nodes_attrs.insert(id.into(), attrs);
        } else {
            self.existing_nodes_attrs.insert(id.into(), attrs);
        }
        Ok(())
    }

    pub fn set_node_attribute(
        &mut self,
        id: NodeId,
        attr_id: u16,
        value: Value,
    ) -> Result<(), String> {
        validate_node_property(&value)?;
        let map = if self.created_nodes.contains(id.into()) {
            &mut self.new_nodes_attrs
        } else {
            &mut self.existing_nodes_attrs
        };
        let entry = map.entry(id.into()).or_default();
        match entry.binary_search_by_key(&attr_id, |(k, _)| *k) {
            Ok(pos) => entry[pos].1 = value,
            Err(pos) => entry.insert(pos, (attr_id, value)),
        }
        Ok(())
    }

    pub fn clear_node_attributes(
        &mut self,
        id: NodeId,
    ) {
        self.new_nodes_attrs.remove(&id.into());
        self.existing_nodes_attrs.remove(&id.into());
    }

    #[must_use]
    #[inline]
    pub fn has_node_attrs(&self) -> bool {
        !self.new_nodes_attrs.is_empty() || !self.existing_nodes_attrs.is_empty()
    }

    #[must_use]
    #[inline]
    pub fn has_relationship_attrs(&self) -> bool {
        !self.new_relationships_attrs.is_empty() || !self.existing_relationships_attrs.is_empty()
    }

    #[must_use]
    pub fn get_node_attribute(
        &self,
        id: NodeId,
        attr_id: u16,
    ) -> Option<&Value> {
        if !self.has_node_attrs() {
            return None;
        }
        self.new_nodes_attrs
            .get(&id.into())
            .and_then(|attrs| lookup_sorted(attrs, attr_id))
            .or_else(|| {
                self.existing_nodes_attrs
                    .get(&id.into())
                    .and_then(|attrs| lookup_sorted(attrs, attr_id))
            })
    }

    pub fn update_node_attrs(
        &self,
        id: NodeId,
        attrs: &mut OrderMap<Arc<String>, Value>,
        g: &Graph,
    ) {
        let added = self
            .new_nodes_attrs
            .get(&id.into())
            .or_else(|| self.existing_nodes_attrs.get(&id.into()));
        if let Some(added) = added {
            for (attr_id, value) in added {
                let Some(key) = g.node_attr_name(*attr_id) else {
                    continue;
                };
                if matches!(value, Value::Null) {
                    attrs.remove(&key);
                } else {
                    attrs.insert(key, value.clone());
                }
            }
        }
    }

    pub fn set_node_labels(
        &mut self,
        id: NodeId,
        labels: &OrderSet<LabelId>,
    ) {
        let entry = self.set_labels.entry(id.into()).or_default();
        for label in labels.iter() {
            entry.push(usize::from(*label) as u64);
        }
    }

    pub fn set_nodes_labels(
        &mut self,
        ids: &[NodeId],
        labels: &OrderSet<LabelId>,
    ) {
        for id in ids {
            let entry = self.set_labels.entry((*id).into()).or_default();
            for label in labels.iter() {
                entry.push(usize::from(*label) as u64);
            }
        }
    }

    pub fn remove_node_labels(
        &mut self,
        id: NodeId,
        labels: &[LabelId],
    ) {
        let raw_id: u64 = id.into();
        for label in labels {
            let label_id = usize::from(*label) as u64;
            // Remove from pending set labels
            if let Some(set) = self.set_labels.get_mut(&raw_id) {
                set.retain(|&l| l != label_id);
            }
            self.remove_labels.entry(raw_id).or_default().push(label_id);
        }
    }

    pub fn update_node_labels(
        &self,
        id: NodeId,
        labels: &mut OrderSet<LabelId>,
    ) {
        let raw_id: u64 = id.into();
        if let Some(set) = self.set_labels.get(&raw_id) {
            for &label_id in set {
                labels.insert(LabelId(label_id as usize));
            }
        }
        if let Some(removed) = self.remove_labels.get(&raw_id) {
            for &label_id in removed {
                labels.remove(&LabelId(label_id as usize));
            }
        }
    }

    pub fn deleted_node(
        &mut self,
        id: NodeId,
    ) {
        self.deleted_nodes.insert(id.into());
    }

    /// Delete a pending-created node: mark it deleted, collect its labels and attrs,
    /// and also mark any pending-created relationships connected to it for deletion.
    /// Returns (label_ids, attrs, connected_pending_rels) — the relationships carry
    /// their type and staged attrs so callers can snapshot them for later reads.
    pub fn delete_pending_node(
        &mut self,
        id: NodeId,
    ) -> (
        OrderSet<LabelId>,
        Vec<(u16, Value)>,
        Vec<(
            RelationshipId,
            NodeId,
            NodeId,
            Arc<String>,
            Option<Vec<(u16, Value)>>,
        )>,
    ) {
        self.created_nodes.remove(id.into());
        // Collect pending labels
        let mut label_ids = OrderSet::default();
        self.update_node_labels(id, &mut label_ids);
        self.set_labels.remove(&id.into());

        // Collect pending attrs
        let attrs = self
            .new_nodes_attrs
            .remove(&id.into())
            .or_else(|| self.existing_nodes_attrs.remove(&id.into()))
            .unwrap_or_default();

        let rels = self.remove_pending_relationships_for_node(id);

        (label_ids, attrs, rels)
    }

    /// Remove and return all pending-created relationships incident on the
    /// given node, along with their staged attributes. Also cleans up
    /// `new_relationships_attrs` and `deleted_relationships` entries for
    /// each removed relationship so that `commit()` has no stale state.
    pub fn remove_pending_relationships_for_node(
        &mut self,
        id: NodeId,
    ) -> Vec<(
        RelationshipId,
        NodeId,
        NodeId,
        Arc<String>,
        Option<Vec<(u16, Value)>>,
    )> {
        let mut rels = Vec::new();
        for (type_name, entries) in &self.created_rels_by_type {
            for &(rel_id, from, to) in entries {
                if from == id || to == id {
                    rels.push((rel_id, from, to, type_name.clone()));
                }
            }
        }

        let mut result = Vec::with_capacity(rels.len());
        for (rel_id, from, to, type_name) in rels {
            self.created_rel_types.remove(&rel_id);
            if let Some(entries) = self.created_rels_by_type.get_mut(&type_name) {
                entries.retain(|(rid, _, _)| *rid != rel_id);
            }
            let attrs = self.new_relationships_attrs.remove(&rel_id.into());
            self.deleted_relationships.remove(rel_id.into());
            result.push((rel_id, from, to, type_name, attrs));
        }

        result
    }

    pub fn created_relationship(
        &mut self,
        id: RelationshipId,
        from: NodeId,
        to: NodeId,
        type_name: Arc<String>,
    ) {
        self.created_rels_by_type
            .entry(type_name.clone())
            .or_default()
            .push((id, from, to));
        self.created_rel_types.insert(id, type_name);
    }

    /// Set all attributes for a relationship. `attrs` must be
    /// attribute-id-resolved, sorted by id, and unique.
    pub fn set_relationship_attributes(
        &mut self,
        id: RelationshipId,
        attrs: Vec<(u16, Value)>,
    ) -> Result<(), String> {
        for (_, value) in &attrs {
            validate_relationship_property(value)?;
        }
        // Empty attribute maps from CREATE without `{...}` props would otherwise
        // create an empty pinned cache entry per relationship on commit; skip them.
        if attrs.is_empty() {
            return Ok(());
        }
        debug_assert!(attrs.is_sorted_by_key(|(k, _)| *k));
        if self.created_rel_types.contains_key(&id) {
            self.new_relationships_attrs.insert(id.into(), attrs);
        } else {
            self.existing_relationships_attrs.insert(id.into(), attrs);
        }
        Ok(())
    }

    pub fn set_relationship_attribute(
        &mut self,
        id: RelationshipId,
        attr_id: u16,
        value: Value,
    ) -> Result<(), String> {
        validate_relationship_property(&value)?;
        let map = if self.created_rel_types.contains_key(&id) {
            &mut self.new_relationships_attrs
        } else {
            &mut self.existing_relationships_attrs
        };
        let entry = map.entry(id.into()).or_default();
        match entry.binary_search_by_key(&attr_id, |(k, _)| *k) {
            Ok(pos) => entry[pos].1 = value,
            Err(pos) => entry.insert(pos, (attr_id, value)),
        }
        Ok(())
    }

    #[must_use]
    pub fn get_relationship_attribute(
        &self,
        id: RelationshipId,
        attr_id: u16,
    ) -> Option<&Value> {
        if !self.has_relationship_attrs() {
            return None;
        }
        self.new_relationships_attrs
            .get(&id.into())
            .and_then(|attrs| lookup_sorted(attrs, attr_id))
            .or_else(|| {
                self.existing_relationships_attrs
                    .get(&id.into())
                    .and_then(|attrs| lookup_sorted(attrs, attr_id))
            })
    }

    pub fn update_relationship_attrs(
        &self,
        id: RelationshipId,
        attrs: &mut OrderMap<Arc<String>, Value>,
        g: &Graph,
    ) {
        let added = self
            .new_relationships_attrs
            .get(&id.into())
            .or_else(|| self.existing_relationships_attrs.get(&id.into()));
        if let Some(added) = added {
            for (attr_id, value) in added {
                let Some(key) = g.rel_attr_name(*attr_id) else {
                    continue;
                };
                if matches!(value, Value::Null) {
                    attrs.remove(&key);
                } else {
                    attrs.insert(key, value.clone());
                }
            }
        }
    }

    pub fn deleted_relationship(
        &mut self,
        id: RelationshipId,
    ) {
        self.deleted_relationships.insert(id.into());
    }

    pub fn deleted_relationships_bulk(
        &mut self,
        rels: &[RelationshipId],
    ) {
        for &id in rels {
            self.deleted_relationships.insert(id.into());
        }
    }

    #[must_use]
    pub fn get_relationship_type(
        &self,
        id: RelationshipId,
    ) -> Option<Arc<String>> {
        self.created_rel_types.get(&id).cloned()
    }

    #[must_use]
    pub fn is_node_created(
        &self,
        id: NodeId,
    ) -> bool {
        self.created_nodes.contains(id.into())
    }

    /// Returns pending-created node IDs that have ALL of the given labels.
    /// When `label_ids` is empty, returns all pending-created nodes.
    #[must_use]
    pub fn get_pending_nodes_with_labels(
        &self,
        label_ids: &[LabelId],
    ) -> RoaringTreemap {
        if label_ids.is_empty() {
            return self.created_nodes.clone();
        }
        let label_ids_u64: Vec<u64> = label_ids.iter().map(|l| usize::from(*l) as u64).collect();
        let mut result = RoaringTreemap::new();
        for (&node_id, node_labels) in &self.set_labels {
            if label_ids_u64.iter().all(|lid| node_labels.contains(lid)) {
                result.insert(node_id);
            }
        }
        result
    }

    /// Returns existing (non-created) node IDs that have pending label REMOVEs
    /// for ANY of the given label_ids.
    #[must_use]
    pub fn nodes_with_pending_label_removes(
        &self,
        label_ids: &[LabelId],
    ) -> RoaringTreemap {
        if label_ids.is_empty() {
            return RoaringTreemap::new();
        }
        let label_ids_u64: Vec<u64> = label_ids.iter().map(|l| usize::from(*l) as u64).collect();
        let mut result = RoaringTreemap::new();
        for (&node_id, removed_labels) in &self.remove_labels {
            if label_ids_u64.iter().any(|lid| removed_labels.contains(lid)) {
                result.insert(node_id);
            }
        }
        result
    }

    #[must_use]
    pub fn is_relationship_created(
        &self,
        id: RelationshipId,
    ) -> bool {
        self.created_rel_types.contains_key(&id)
    }

    /// Returns (src, dst) for a pending-created relationship, or None if not found.
    #[must_use]
    pub fn get_created_relationship_endpoints(
        &self,
        id: RelationshipId,
    ) -> Option<(NodeId, NodeId)> {
        let type_name = self.created_rel_types.get(&id)?;
        self.created_rels_by_type
            .get(type_name)?
            .iter()
            .find(|(rid, _, _)| *rid == id)
            .map(|&(_, from, to)| (from, to))
    }

    #[must_use]
    pub fn has_deleted_nodes(&self) -> bool {
        !self.deleted_nodes.is_empty()
    }

    #[must_use]
    pub fn has_deleted_relationships(&self) -> bool {
        !self.deleted_relationships.is_empty()
    }

    #[must_use]
    pub fn has_created_relationships(&self) -> bool {
        !self.created_rel_types.is_empty()
    }

    #[must_use]
    pub fn is_node_deleted(
        &self,
        id: NodeId,
    ) -> bool {
        self.deleted_nodes.contains(id.into())
    }

    /// Returns a clone of the pending-deleted nodes bitmap.
    #[must_use]
    pub fn deleted_nodes(&self) -> RoaringTreemap {
        self.deleted_nodes.clone()
    }

    #[must_use]
    pub fn is_relationship_deleted(
        &self,
        id: RelationshipId,
    ) -> bool {
        self.deleted_relationships.contains(id.into())
    }

    /// Count pending-created relationships whose destination is `node_id` and
    /// whose type name matches one of `types` (or all if `types` is empty).
    #[must_use]
    pub fn pending_indegree(
        &self,
        node_id: NodeId,
        types: &[Arc<String>],
    ) -> usize {
        if types.is_empty() {
            self.created_rels_by_type
                .values()
                .flat_map(|v| v.iter())
                .filter(|(_, _, to)| *to == node_id)
                .count()
        } else {
            types
                .iter()
                .filter_map(|t| self.created_rels_by_type.get(t))
                .flat_map(|v| v.iter())
                .filter(|(_, _, to)| *to == node_id)
                .count()
        }
    }

    /// Count pending-created relationships whose source is `node_id` and
    /// whose type name matches one of `types` (or all if `types` is empty).
    #[must_use]
    pub fn pending_outdegree(
        &self,
        node_id: NodeId,
        types: &[Arc<String>],
    ) -> usize {
        if types.is_empty() {
            self.created_rels_by_type
                .values()
                .flat_map(|v| v.iter())
                .filter(|(_, from, _)| *from == node_id)
                .count()
        } else {
            types
                .iter()
                .filter_map(|t| self.created_rels_by_type.get(t))
                .flat_map(|v| v.iter())
                .filter(|(_, from, _)| *from == node_id)
                .count()
        }
    }

    /// Count pending-deleted relationships whose destination is `node_id` and
    /// whose type name matches one of `types` (or all if `types` is empty).
    /// Requires access to the graph to resolve relationship type IDs.
    #[must_use]
    pub fn pending_deleted_indegree(
        &self,
        node_id: NodeId,
        types: &[Arc<String>],
        g: &Graph,
    ) -> usize {
        self.deleted_relationships
            .iter()
            .filter(|rel_id| {
                let (_from, to) = g.get_relationship_endpoints(RelationshipId::from(*rel_id));
                to == node_id
                    && (types.is_empty()
                        || g.get_type(g.get_relationship_type_id(RelationshipId::from(*rel_id)))
                            .is_some_and(|t| types.contains(&t)))
            })
            .count()
    }

    /// Count pending-deleted relationships whose source is `node_id` and
    /// whose type name matches one of `types` (or all if `types` is empty).
    /// Requires access to the graph to resolve relationship type IDs.
    #[must_use]
    pub fn pending_deleted_outdegree(
        &self,
        node_id: NodeId,
        types: &[Arc<String>],
        g: &Graph,
    ) -> usize {
        self.deleted_relationships
            .iter()
            .filter(|rel_id| {
                let (from, _to) = g.get_relationship_endpoints(RelationshipId::from(*rel_id));
                from == node_id
                    && (types.is_empty()
                        || g.get_type(g.get_relationship_type_id(RelationshipId::from(*rel_id)))
                            .is_some_and(|t| types.contains(&t)))
            })
            .count()
    }

    pub fn commit(
        &mut self,
        g: &AtomicRefCell<Graph>,
        stats: &RefCell<QueryStatistics>,
    ) -> Result<(), String> {
        if !self.created_nodes.is_empty() {
            stats.borrow_mut().nodes_created += self.created_nodes.len();
            g.borrow_mut().create_nodes(&self.created_nodes);
        }
        if !self.created_rel_types.is_empty() {
            stats.borrow_mut().relationships_created += self.created_rel_types.len();
            let mut g = g.borrow_mut();
            for (type_name, rel_ids) in &self.created_rels_by_type {
                let mut srcs = Vec::with_capacity(rel_ids.len());
                let mut dsts = Vec::with_capacity(rel_ids.len());
                let mut ids = Vec::with_capacity(rel_ids.len());
                for &(rel_id, from, to) in rel_ids {
                    srcs.push(from.into());
                    dsts.push(to.into());
                    ids.push(rel_id.into());
                }
                g.create_relationships_bulk(type_name, &srcs, &dsts, &ids);
            }
        }
        if !self.set_labels.is_empty() {
            let (rows, cols) = flatten_label_map(&self.set_labels);
            // Pairs for nodes created in this transaction cannot already be
            // committed (fresh ids; a reclaimed id's stale entries carry dm
            // tombstones), so a pure-create batch takes the unchecked insert.
            let all_new = self
                .set_labels
                .keys()
                .all(|id| self.created_nodes.contains(*id));
            g.borrow_mut()
                .set_nodes_labels_bulk(&rows, &cols, &mut self.index_add_docs, all_new);
        }
        if !self.remove_labels.is_empty() {
            let (rows, cols) = flatten_label_map(&self.remove_labels);
            stats.borrow_mut().labels_removed += rows.len();
            g.borrow_mut()
                .remove_nodes_labels(&rows, &cols, &mut self.index_remove_docs);
        }
        if !self.new_nodes_attrs.is_empty() || !self.existing_nodes_attrs.is_empty() {
            let mut g = g.borrow_mut();
            if !self.new_nodes_attrs.is_empty() {
                let nset = g.import_node_attrs(
                    &self.new_nodes_attrs,
                    &self.set_labels,
                    &mut self.index_add_docs,
                );
                stats.borrow_mut().properties_set += nset;
            }
            if !self.existing_nodes_attrs.is_empty() {
                let (nremoved, nset) =
                    g.set_nodes_attributes(&self.existing_nodes_attrs, &mut self.index_add_docs)?;
                let mut s = stats.borrow_mut();
                s.properties_set += nset;
                s.properties_removed += nremoved;
            }
        }

        if !self.new_relationships_attrs.is_empty() || !self.existing_relationships_attrs.is_empty()
        {
            let mut g = g.borrow_mut();
            if !self.new_relationships_attrs.is_empty() {
                let nset = g.import_relationship_attrs(
                    &self.new_relationships_attrs,
                    &mut self.index_add_edge_docs,
                );
                stats.borrow_mut().properties_set += nset;
            }
            if !self.existing_relationships_attrs.is_empty() {
                let (nremoved, nset) = g.set_relationships_attributes(
                    &self.existing_relationships_attrs,
                    &mut self.index_add_edge_docs,
                )?;
                let mut s = stats.borrow_mut();
                s.properties_set += nset;
                s.properties_removed += nremoved;
            }
        }
        if !self.deleted_nodes.is_empty() {
            stats.borrow_mut().nodes_deleted += self.deleted_nodes.len();
            g.borrow_mut()
                .delete_nodes(&self.deleted_nodes, &mut self.index_remove_docs)?;
        }
        // Take relationship deletions BEFORE implicit edge processing
        // so we can pass them to delete_implicit_edges for dedup.
        let explicit_rels = std::mem::take(&mut self.deleted_relationships);

        // Bulk cascade-delete edges for implicitly deleted nodes.
        // This must run after delete_nodes so that node matrices are already
        // cleaned up, and before delete_relationships so explicit edges are
        // still tracked separately.
        if !self.deleted_nodes.is_empty() {
            let implicit_edges = g.borrow_mut().delete_implicit_edges(
                &self.deleted_nodes,
                &explicit_rels,
                &mut self.index_remove_edge_docs,
            )?;
            let count = implicit_edges.len();
            stats.borrow_mut().relationships_deleted += count;
            // Record in deleted_relationships so effects buffer can serialize them
            for (rel_id, type_id, from, to) in implicit_edges {
                self.deleted_relationships.insert(u64::from(rel_id));
                self.deleted_endpoints.push((rel_id, type_id, from, to));
            }
        }
        if !explicit_rels.is_empty() {
            let endpoints = g
                .borrow_mut()
                .delete_relationships(&explicit_rels, &mut self.index_remove_edge_docs)?;
            // Use the actually-removed relationships (delete_relationships skips
            // stale/missing ids) for stats and effects/constraint bookkeeping.
            stats.borrow_mut().relationships_deleted += endpoints.len();
            self.deleted_relationships
                .extend(endpoints.iter().map(|(id, _, _, _)| u64::from(*id)));
            self.deleted_endpoints.extend(endpoints);
        }
        // Enforce constraints before accumulating index operations.
        // The constraint checks read the attribute store, which already has
        // the mutations from this transaction (writes apply immediately to
        // the private MVCC graph).
        self.enforce_constraints(g)?;

        // Index operations are deferred — they will be applied only after
        // the full query succeeds to avoid stale RediSearch entries on
        // rollback.

        // Accumulate index operations into deferred fields.
        for (k, v) in self.index_add_docs.drain() {
            self.deferred_index_adds
                .entry(k)
                .or_default()
                .bitor_assign(&v);
        }
        for (k, v) in self.index_remove_docs.drain() {
            self.deferred_index_removes
                .entry(k)
                .or_default()
                .bitor_assign(&v);
        }
        for (k, v) in self.index_add_edge_docs.drain() {
            self.deferred_edge_index_adds
                .entry(k)
                .or_default()
                .bitor_assign(&v);
        }
        for (k, v) in self.index_remove_edge_docs.drain() {
            self.deferred_edge_index_removes
                .entry(k)
                .or_default()
                .extend(v);
        }

        Ok(())
    }

    /// Enforce graph constraints on all entities affected by this transaction.
    fn enforce_constraints(
        &self,
        g: &AtomicRefCell<Graph>,
    ) -> Result<(), String> {
        let g = g.borrow();
        let constraints = g.constraints();
        if constraints.is_empty() {
            return Ok(());
        }

        // Collect affected node IDs and their labels
        // Affected = created + had attributes modified + had labels added
        let mut affected_node_ids = RoaringTreemap::new();
        affected_node_ids |= &self.created_nodes;
        for &id in self.new_nodes_attrs.keys() {
            affected_node_ids.insert(id);
        }
        for &id in self.existing_nodes_attrs.keys() {
            affected_node_ids.insert(id);
        }
        for &id in self.set_labels.keys() {
            affected_node_ids.insert(id);
        }

        // Collect affected edge IDs
        let mut affected_edge_ids = RoaringTreemap::new();
        for rels in self.created_rels_by_type.values() {
            for &(rel_id, _, _) in rels {
                affected_edge_ids.insert(rel_id.into());
            }
        }
        for &id in self.new_relationships_attrs.keys() {
            affected_edge_ids.insert(id);
        }
        for &id in self.existing_relationships_attrs.keys() {
            affected_edge_ids.insert(id);
        }

        // Remove deleted entities from affected sets
        for id in &self.deleted_nodes {
            affected_node_ids.remove(id);
        }
        for rel_id in &self.deleted_relationships {
            affected_edge_ids.remove(rel_id);
        }

        // Check each OPERATIONAL constraint
        for constraint in constraints {
            if constraint.status != ConstraintStatus::Operational {
                continue;
            }

            match constraint.entity_type {
                EntityType::Node => {
                    self.check_node_constraint(&g, constraint, &affected_node_ids)?;
                }
                EntityType::Relationship => {
                    self.check_edge_constraint(&g, constraint, &affected_edge_ids)?;
                }
            }
        }

        Ok(())
    }

    /// Label check for constraint enforcement that answers from this
    /// transaction's own bookkeeping first. Reading the label matrix here
    /// would force a pending-tuple materialization of its delta on every
    /// commit (`O(|delta|)` per write query, quadratic between folds) —
    /// measured as the dominant cost of small repeated creates. Mirrors
    /// [`Self::update_node_labels`] semantics: a removed label wins over a
    /// pending set.
    fn constraint_node_has_label(
        &self,
        g: &Graph,
        node_id: u64,
        label_id: u64,
    ) -> bool {
        if let Some(removed) = self.remove_labels.get(&node_id)
            && removed.contains(&label_id)
        {
            return false;
        }
        if let Some(set) = self.set_labels.get(&node_id)
            && set.contains(&label_id)
        {
            return true;
        }
        if self.created_nodes.contains(node_id) {
            // Labels of nodes created this transaction live entirely in
            // set_labels — no need to touch the graph.
            return false;
        }
        g.node_has_label_id(node_id.into(), LabelId(label_id as usize))
    }

    fn check_node_constraint(
        &self,
        g: &Graph,
        constraint: &crate::graph::constraint::Constraint,
        affected_node_ids: &RoaringTreemap,
    ) -> Result<(), String> {
        let label = &constraint.label;
        let Some(label_id) = g.get_label_id(label) else {
            // Label doesn't exist yet — no node can carry it.
            return Ok(());
        };
        let label_id = usize::from(label_id) as u64;

        for node_id in affected_node_ids {
            // Check if this node has the constrained label
            if !self.constraint_node_has_label(g, node_id, label_id) {
                continue;
            }

            match constraint.ct {
                ConstraintType::Mandatory => {
                    for prop in &constraint.properties {
                        let has_prop = g
                            .get_node_attribute(node_id.into(), prop)
                            .is_some_and(|val| !matches!(val, Value::Null));
                        if !has_prop {
                            return Err(format!(
                                "mandatory constraint violation: node with label {label} missing property {prop}"
                            ));
                        }
                    }
                }
                ConstraintType::Unique => {
                    let key = Graph::build_composite_key(&constraint.properties, |prop| {
                        g.get_node_attribute(node_id.into(), prop)
                    });
                    if key.is_empty() {
                        continue; // All NULL → no violation
                    }

                    // Build a set of all existing keys for this label in one pass
                    if let Some(lm) = g.get_label_matrix(label) {
                        let mut seen: FxHashMap<Vec<u8>, u64> = FxHashMap::default();
                        for (other_id, _) in lm.iter(0, u64::MAX) {
                            let other_key =
                                Graph::build_composite_key(&constraint.properties, |prop| {
                                    g.get_node_attribute(other_id.into(), prop)
                                });
                            if other_key.is_empty() {
                                continue;
                            }
                            if let Some(&existing_id) = seen.get(&other_key)
                                && existing_id != other_id
                            {
                                return Err(format!(
                                    "unique constraint violation on node of type {label}"
                                ));
                            }
                            seen.insert(other_key, other_id);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn check_edge_constraint(
        &self,
        g: &Graph,
        constraint: &crate::graph::constraint::Constraint,
        affected_edge_ids: &RoaringTreemap,
    ) -> Result<(), String> {
        let type_name = &constraint.label;

        for edge_id in affected_edge_ids {
            // Edges created this transaction resolve their type from
            // Pending's reverse index — no relationship-matrix read, which
            // would materialize the delta's pending tuples on every commit.
            // An edge's type never changes, so created under a different
            // type means not a member.
            let has_type = match self.created_rel_types.get(&edge_id.into()) {
                Some(created_type) => created_type.as_str() == type_name.as_str(),
                None => g.edge_has_type(edge_id.into(), type_name),
            };
            if !has_type {
                continue;
            }

            match constraint.ct {
                ConstraintType::Mandatory => {
                    for prop in &constraint.properties {
                        let has_prop = g
                            .get_relationship_attribute(edge_id.into(), prop)
                            .is_some_and(|val| !matches!(val, Value::Null));
                        if !has_prop {
                            return Err(format!(
                                "mandatory constraint violation: edge with relationship-type {type_name} missing property {prop}"
                            ));
                        }
                    }
                }
                ConstraintType::Unique => {
                    let key = Graph::build_composite_key(&constraint.properties, |prop| {
                        g.get_relationship_attribute(edge_id.into(), prop)
                    });
                    if key.is_empty() {
                        continue;
                    }

                    // Build a set of all existing keys for this type in one pass
                    if let Some(tensor) = g.get_relationship_matrix(type_name) {
                        let mut seen: FxHashMap<Vec<u8>, u64> = FxHashMap::default();
                        for (_, _, other_eid) in tensor.iter(0, u64::MAX, false) {
                            let other_key =
                                Graph::build_composite_key(&constraint.properties, |prop| {
                                    g.get_relationship_attribute(other_eid.into(), prop)
                                });
                            if other_key.is_empty() {
                                continue;
                            }
                            if let Some(&existing_id) = seen.get(&other_key)
                                && existing_id != other_eid
                            {
                                return Err(format!(
                                    "unique constraint violation, on edge of relationship-type {type_name}"
                                ));
                            }
                            seen.insert(other_key, other_eid);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Take the accumulated index document changes, leaving pending empty.
    pub fn take_deferred_indexes(&mut self) -> DeferredIndexes {
        DeferredIndexes {
            node_adds: std::mem::take(&mut self.deferred_index_adds),
            node_removes: std::mem::take(&mut self.deferred_index_removes),
            edge_adds: std::mem::take(&mut self.deferred_edge_index_adds),
            edge_removes: std::mem::take(&mut self.deferred_edge_index_removes),
        }
    }

    /// Undo the index documents earlier `Commit`s published, after this query
    /// failed, by re-synchronising them against committed state.
    ///
    /// Mirrors C's undo log (`_UndoLog_Rollback_Update_Entity`): a failed *create*
    /// loses its document, a failed *update* is re-indexed from the entity's
    /// previous values — deleting it instead would drop a live entity out of the
    /// index. No undo log is needed for that, because MVCC still holds the previous
    /// state: `committed` *is* the old value, and re-adding rewrites the document.
    ///
    /// `private` is the version this query built and is about to discard; a removed
    /// edge's document key comes from there, since committed state no longer has the
    /// edge at all.
    ///
    /// Writer mode only, and before the write lock is released.
    pub fn resync_published_indexes(
        &mut self,
        committed: &AtomicRefCell<Graph>,
        private: &AtomicRefCell<Graph>,
    ) {
        let (nodes, edges) = std::mem::take(&mut self.published).ids_by_slot();
        if nodes.is_empty() && edges.is_empty() {
            return;
        }

        let mut node_adds: FxHashMap<u64, RoaringTreemap> = FxHashMap::default();
        let mut node_removes: FxHashMap<u64, RoaringTreemap> = FxHashMap::default();
        let mut edge_adds: FxHashMap<u64, RoaringTreemap> = FxHashMap::default();
        let mut edge_removes: FxHashMap<u64, FxHashMap<u64, (u64, u64)>> = FxHashMap::default();
        {
            let g = committed.borrow();
            for (slot, ids) in &nodes {
                for id in ids {
                    if g.node_has_label_id(NodeId::from(id), LabelId(*slot as usize)) {
                        // Still live: rewrite the document from committed values.
                        node_adds.entry(*slot).or_default().insert(id);
                    } else {
                        // Never existed outside the discarded version.
                        node_removes.entry(*slot).or_default().insert(id);
                    }
                }
            }
            let p = private.borrow();
            for (slot, ids) in &edges {
                for id in ids {
                    if g.endpoints_for_edge(id).is_some() {
                        edge_adds.entry(*slot).or_default().insert(id);
                    } else if let Some(endpoints) = p.endpoints_for_edge(id) {
                        // The doc key is `(src, dst, edge_id)` as written, and only the
                        // discarded version still knows the endpoints.
                        edge_removes.entry(*slot).or_default().insert(id, endpoints);
                    }
                }
            }
        }

        // Indexer locks *before* the graph borrow — the order `populate_index_batch`
        // uses, and the only one that keeps this mutable borrow of the published
        // version from colliding with a populate batch reading it (`AtomicRefCell`
        // reports that collision by panicking). Concurrent *readers* are excluded
        // instead by the write lock this query holds.
        let (node_lock, edge_lock) = committed.borrow().index_locks();
        let node_guard = node_lock.lock();
        let edge_guard = edge_lock.lock();
        let mut g = committed.borrow_mut();
        g.commit_index_locked(&node_guard, &mut node_adds, &mut node_removes);
        g.commit_edge_index_locked(&edge_guard, &mut edge_adds, &mut edge_removes);
    }

    /// Write this `Commit`'s index documents to RediSearch. Writer mode only.
    pub fn commit_deferred_indexes(
        &mut self,
        g: &AtomicRefCell<Graph>,
    ) {
        let mut deferred = self.take_deferred_indexes();
        // Fold the batch into the published log *before* writing it, because `commit`
        // drains the batch. Removals matter as much as additions: a rolled-back
        // DELETE must get its document back, or the entity silently disappears from
        // the index.
        self.published.merge(&deferred);
        deferred.commit(g);
    }

    /// Clear all pending mutation state.
    ///
    /// Runs after every `Commit`, so it must NOT touch `published` — a later failure
    /// has to undo the documents *all* of this query's `Commit`s published, not just
    /// the last one's.
    pub fn clear(&mut self) {
        // Dropping millions of per-entity Vec allocations is O(n) frees and
        // stalls the serialized write thread; move large maps to a background
        // thread and let it pay the deallocation cost.
        const OFFLOAD_THRESHOLD: usize = 4096;
        let big_entries = self.new_nodes_attrs.len()
            + self.existing_nodes_attrs.len()
            + self.new_relationships_attrs.len()
            + self.existing_relationships_attrs.len()
            + self.set_labels.len()
            + self.remove_labels.len()
            + self
                .created_rels_by_type
                .values()
                .map(Vec::len)
                .sum::<usize>();
        if big_entries >= OFFLOAD_THRESHOLD {
            let maps = (
                std::mem::take(&mut self.new_nodes_attrs),
                std::mem::take(&mut self.existing_nodes_attrs),
                std::mem::take(&mut self.new_relationships_attrs),
                std::mem::take(&mut self.existing_relationships_attrs),
                std::mem::take(&mut self.set_labels),
                std::mem::take(&mut self.remove_labels),
                std::mem::take(&mut self.created_rels_by_type),
            );
            std::thread::spawn(move || drop(maps));
        } else {
            self.new_nodes_attrs.clear();
            self.existing_nodes_attrs.clear();
            self.new_relationships_attrs.clear();
            self.existing_relationships_attrs.clear();
            self.set_labels.clear();
            self.remove_labels.clear();
            self.created_rels_by_type.clear();
        }
        self.created_nodes.clear();
        self.created_rel_types.clear();
        self.deleted_nodes.clear();
        self.deleted_relationships.clear();
        self.deleted_endpoints.clear();
        self.index_add_docs.clear();
        self.index_remove_docs.clear();
        self.index_add_edge_docs.clear();
        self.index_remove_edge_docs.clear();
    }

    /// Returns the number of effects (operations) tracked in this Pending.
    #[must_use]
    pub fn effects_count(&self) -> u64 {
        self.created_nodes.len()
            + self.created_rel_types.len() as u64
            + self.deleted_nodes.len()
            + self.deleted_relationships.len()
            + self.new_nodes_attrs.len() as u64
            + self.existing_nodes_attrs.len() as u64
            + self.new_relationships_attrs.len() as u64
            + self.existing_relationships_attrs.len() as u64
            + self
                .set_labels
                .values()
                .map(|v| v.len() as u64)
                .sum::<u64>()
            + self
                .remove_labels
                .values()
                .map(|v| v.len() as u64)
                .sum::<u64>()
    }

    /// Build a binary effects buffer from the accumulated mutations.
    /// Must be called before `clear()` resets the pending data.
    /// Appends to an existing buffer if provided, so multiple commits
    /// in the same query accumulate into a single effects buffer.
    /// Returns the number of effect records written.
    pub fn build_effects_buffer(
        &self,
        g: &AtomicRefCell<Graph>,
        buf: &mut Vec<u8>,
    ) -> u64 {
        let mut n_effects = 0u64;

        // Pre-allocate buffer: entity headers plus ~12 bytes per attribute
        // (2-byte attribute id + tagged value payload).
        let attr_bytes: usize = self
            .new_nodes_attrs
            .values()
            .chain(self.existing_nodes_attrs.values())
            .chain(self.new_relationships_attrs.values())
            .chain(self.existing_relationships_attrs.values())
            .map(|m| m.len() * 12)
            .sum();
        let estimated_bytes = (self.created_nodes.len() as usize) * 15
            + self.created_rel_types.len() * 30
            + (self.deleted_nodes.len() as usize) * 10
            + (self.deleted_relationships.len() as usize) * 25
            + attr_bytes;
        buf.reserve(estimated_bytes);

        // Version header (only write once at the start)
        if buf.is_empty() {
            buf.push(EFFECTS_VERSION);
        }

        // --- Schema additions (new labels, relationship types) ---
        {
            let graph = g.borrow();
            let labels = graph.get_labels();
            for label in labels.iter().skip(self.schema_label_count) {
                buf.push(EFFECT_ADD_SCHEMA);
                buf.push(SCHEMA_NODE_LABEL);
                write_string(buf, label);
                n_effects += 1;
            }
            let types = graph.get_types();
            for rel_type in types.iter().skip(self.schema_rel_type_count) {
                buf.push(EFFECT_ADD_SCHEMA);
                buf.push(SCHEMA_REL_TYPE);
                write_string(buf, rel_type);
                n_effects += 1;
            }

            // --- Attribute additions (new node/rel attribute names) ---
            let node_attrs = graph.get_node_attribute_names();
            for attr in node_attrs.iter().skip(self.schema_node_attr_count) {
                buf.push(EFFECT_ADD_ATTRIBUTE);
                buf.push(ATTR_NODE);
                write_string(buf, attr);
                n_effects += 1;
            }
            let rel_attrs = graph.get_relationship_attribute_names();
            for attr in rel_attrs.iter().skip(self.schema_rel_attr_count) {
                buf.push(EFFECT_ADD_ATTRIBUTE);
                buf.push(ATTR_REL);
                write_string(buf, attr);
                n_effects += 1;
            }
        }

        // Attribute keys, label ids, and relationship type ids are encoded
        // as u16 ids; the id → name mapping is established on the replica by
        // the EFFECT_ADD_SCHEMA / EFFECT_ADD_ATTRIBUTE records above (and by
        // in-order replay of earlier queries), which mirror the master's
        // registration order exactly.

        // --- Created nodes ---
        for node_id in &self.created_nodes {
            buf.push(EFFECT_CREATE_NODE);
            buf.extend_from_slice(&node_id.to_le_bytes());

            // Labels
            if let Some(label_ids) = self.set_labels.get(&node_id) {
                write_u16(buf, label_ids.len() as u16);
                for &label_id in label_ids {
                    write_u16(buf, label_id as u16);
                }
            } else {
                write_u16(buf, 0);
            }

            // Attributes
            if let Some(attrs) = self.new_nodes_attrs.get(&node_id) {
                write_u16(buf, attrs.len() as u16);
                for (attr_id, value) in attrs {
                    write_u16(buf, *attr_id);
                    write_value(buf, value);
                }
            } else {
                write_u16(buf, 0);
            }
            n_effects += 1;
        }

        // --- Created relationships ---
        if !self.created_rels_by_type.is_empty() {
            let graph = g.borrow();
            for (type_name, entries) in &self.created_rels_by_type {
                let type_id = graph
                    .get_type_id(type_name)
                    .expect("created relationship type must be registered")
                    .0 as u16;
                for &(rel_id, from, to) in entries {
                    buf.push(EFFECT_CREATE_EDGE);
                    buf.extend_from_slice(&u64::from(rel_id).to_le_bytes());
                    buf.extend_from_slice(&u64::from(from).to_le_bytes());
                    buf.extend_from_slice(&u64::from(to).to_le_bytes());
                    write_u16(buf, type_id);

                    if let Some(attrs) = self.new_relationships_attrs.get(&u64::from(rel_id)) {
                        write_u16(buf, attrs.len() as u16);
                        for (attr_id, value) in attrs {
                            write_u16(buf, *attr_id);
                            write_value(buf, value);
                        }
                    } else {
                        write_u16(buf, 0);
                    }
                    n_effects += 1;
                }
            }
        }

        // --- Updated node attributes (existing nodes only) ---
        for (node_id, attrs) in &self.existing_nodes_attrs {
            buf.push(EFFECT_UPDATE_NODE);
            buf.extend_from_slice(&node_id.to_le_bytes());
            write_u16(buf, attrs.len() as u16);
            for (attr_id, value) in attrs {
                write_u16(buf, *attr_id);
                write_value(buf, value);
            }
            n_effects += 1;
        }

        // --- Updated relationship attributes (existing rels only) ---
        for (rel_id, attrs) in &self.existing_relationships_attrs {
            buf.push(EFFECT_UPDATE_EDGE);
            buf.extend_from_slice(&rel_id.to_le_bytes());
            write_u16(buf, attrs.len() as u16);
            for (attr_id, value) in attrs {
                write_u16(buf, *attr_id);
                write_value(buf, value);
            }
            n_effects += 1;
        }

        // --- Set labels (non-created nodes only) ---
        for (&node_id, label_ids) in &self.set_labels {
            if !self.created_nodes.contains(node_id) {
                buf.push(EFFECT_SET_LABELS);
                buf.extend_from_slice(&node_id.to_le_bytes());
                write_u16(buf, label_ids.len() as u16);
                for &label_id in label_ids {
                    write_u16(buf, label_id as u16);
                }
                n_effects += 1;
            }
        }

        // --- Remove labels ---
        for (&node_id, label_ids) in &self.remove_labels {
            buf.push(EFFECT_REMOVE_LABELS);
            buf.extend_from_slice(&node_id.to_le_bytes());
            write_u16(buf, label_ids.len() as u16);
            for &label_id in label_ids {
                write_u16(buf, label_id as u16);
            }
            n_effects += 1;
        }

        // --- Deleted relationships (before nodes, so replica removes edges first) ---
        for &(rel_id, _type_id, from, to) in &self.deleted_endpoints {
            buf.push(EFFECT_DELETE_EDGE);
            buf.extend_from_slice(&u64::from(rel_id).to_le_bytes());
            buf.extend_from_slice(&u64::from(from).to_le_bytes());
            buf.extend_from_slice(&u64::from(to).to_le_bytes());
            n_effects += 1;
        }

        // --- Deleted nodes ---
        for node_id in &self.deleted_nodes {
            buf.push(EFFECT_DELETE_NODE);
            buf.extend_from_slice(&node_id.to_le_bytes());
            n_effects += 1;
        }

        n_effects
    }

    /// Build a **v3** effects payload: one record per `(opcode, partition key)`
    /// instead of one per entity.
    ///
    /// Not yet wired to replication — [`Self::build_effects_buffer`] still owns
    /// that. Both exist side by side until the C engine can read v3, because a
    /// master that emits v3 to a v2 replica is a silent data-loss bug, not a
    /// degraded mode.
    ///
    /// The grouping is cheap because `Pending` already stores most of it:
    /// `created_nodes` is a `RoaringTreemap`, so ids arrive ascending for free;
    /// `created_rels_by_type` is already partitioned by relationship type; and
    /// the attribute vectors are already "sorted by id, unique", which is
    /// exactly the shape key.
    pub fn build_effects_buffer_v3(
        &self,
        g: &AtomicRefCell<Graph>,
        buf: &mut Vec<u8>,
    ) -> u64 {
        use crate::runtime::effects_v3 as v3;

        let mut n_effects = 0u64;
        if buf.is_empty() {
            buf.push(v3::EFFECTS_VERSION);
        }

        // --- Schema and attribute additions, each carrying its id ---
        //
        // These must precede every record that references the ids they
        // introduce. That ordering is normative in v3, not incidental: the ids
        // on the wire are only meaningful once the replica has agreed on the
        // numbering that produced them.
        {
            let graph = g.borrow();
            for (offset, label) in graph
                .get_labels()
                .iter()
                .enumerate()
                .skip(self.schema_label_count)
            {
                v3::write_add_schema(buf, v3::SCHEMA_NODE, offset as i32, label);
                n_effects += 1;
            }
            for (offset, rel_type) in graph
                .get_types()
                .iter()
                .enumerate()
                .skip(self.schema_rel_type_count)
            {
                v3::write_add_schema(buf, v3::SCHEMA_EDGE, offset as i32, rel_type);
                n_effects += 1;
            }
            // Once, not once per entity kind. #2459 unified the two attribute
            // dictionaries, so `get_node_attribute_names` and
            // `get_relationship_attribute_names` return the same store — and
            // unlike v2's, v3's ADD_ATTRIBUTE has no node/relationship
            // discriminator, so emitting both loops would send every new
            // attribute twice under the same id.
            for (offset, attr) in graph
                .get_node_attribute_names()
                .iter()
                .enumerate()
                .skip(self.schema_node_attr_count)
            {
                v3::write_add_attribute(buf, offset as u16, attr);
                n_effects += 1;
            }
        }

        // --- Created nodes, grouped by (labels, attribute ids) ---
        n_effects += self.emit_created_nodes_v3(buf);

        // --- Created relationships, grouped by (type, attribute ids) ---
        n_effects += self.emit_created_edges_v3(g, buf);

        // --- Property updates on entities that already existed ---
        n_effects += emit_updates_v3(buf, &self.existing_nodes_attrs, v3::GraphEntity::Node);
        n_effects += emit_updates_v3(
            buf,
            &self.existing_relationships_attrs,
            v3::GraphEntity::Edge,
        );

        // --- Label changes on nodes this query did not create ---
        // A created node carries its labels in its CREATE_NODE record already.
        n_effects += emit_labels_v3(buf, &self.set_labels, Some(&self.created_nodes), true);
        n_effects += emit_labels_v3(buf, &self.remove_labels, None, false);

        // --- Deletions: edges before nodes, so the replica unhooks first ---
        n_effects += self.emit_deleted_edges_v3(buf);
        if !self.deleted_nodes.is_empty() {
            let ids: Vec<u64> = self.deleted_nodes.iter().collect();
            v3::write_delete_node(buf, &ids);
            n_effects += 1;
        }

        n_effects
    }

    fn emit_created_nodes_v3(
        &self,
        buf: &mut Vec<u8>,
    ) -> u64 {
        use crate::runtime::effects_v3 as v3;

        // Shape key: the node's labels and its attribute ids. Labels are sorted
        // because `set_labels` keeps caller order — [7,8] and [8,7] are the same
        // shape, and emitting them ascending is what makes two engines that
        // agree on the set produce the same bytes.
        let mut groups: FxHashMap<(Vec<i32>, Vec<u16>), Vec<u64>> = FxHashMap::default();
        for id in &self.created_nodes {
            let mut labels: Vec<i32> = self
                .set_labels
                .get(&id)
                .map(|l| l.iter().map(|&v| v as i32).collect())
                .unwrap_or_default();
            labels.sort_unstable();
            labels.dedup();
            let attr_ids: Vec<u16> = self
                .new_nodes_attrs
                .get(&id)
                .map(|a| a.iter().map(|(aid, _)| *aid).collect())
                .unwrap_or_default();
            groups.entry((labels, attr_ids)).or_default().push(id);
        }

        let mut n = 0;
        for ((labels, attr_ids), ids) in sorted_groups(groups) {
            let rows = gather_rows(&ids, &attr_ids, &self.new_nodes_attrs);
            v3::write_create_node(buf, &ids, &labels, &attr_ids, &rows);
            n += 1;
        }
        n
    }

    fn emit_created_edges_v3(
        &self,
        g: &AtomicRefCell<Graph>,
        buf: &mut Vec<u8>,
    ) -> u64 {
        use crate::runtime::effects_v3 as v3;

        if self.created_rels_by_type.is_empty() {
            return 0;
        }
        let graph = g.borrow();
        let mut n = 0;
        for (type_name, entries) in &self.created_rels_by_type {
            let relation_id = graph
                .get_type_id(type_name)
                .expect("created relationship type must be registered")
                .0 as i32;
            // Already partitioned by type; split further by attribute shape.
            let mut groups: FxHashMap<Vec<u16>, Vec<(u64, u64, u64)>> = FxHashMap::default();
            for &(rel_id, from, to) in entries {
                let id = u64::from(rel_id);
                let attr_ids: Vec<u16> = self
                    .new_relationships_attrs
                    .get(&id)
                    .map(|a| a.iter().map(|(aid, _)| *aid).collect())
                    .unwrap_or_default();
                groups
                    .entry(attr_ids)
                    .or_default()
                    .push((id, u64::from(from), u64::from(to)));
            }
            for (attr_ids, mut rows) in sorted_groups(groups) {
                // IdSet requires ascending ids, and the endpoints must stay
                // aligned with them, so sort the triples together.
                rows.sort_unstable();
                let ids: Vec<u64> = rows.iter().map(|r| r.0).collect();
                let src: Vec<u64> = rows.iter().map(|r| r.1).collect();
                let dst: Vec<u64> = rows.iter().map(|r| r.2).collect();
                let values = gather_rows(&ids, &attr_ids, &self.new_relationships_attrs);
                v3::write_create_edge(buf, &ids, relation_id, &src, &dst, &attr_ids, &values);
                n += 1;
            }
        }
        n
    }

    fn emit_deleted_edges_v3(
        &self,
        buf: &mut Vec<u8>,
    ) -> u64 {
        use crate::runtime::effects_v3 as v3;

        let mut groups: FxHashMap<u64, Vec<(u64, u64, u64)>> = FxHashMap::default();
        for &(rel_id, type_id, from, to) in &self.deleted_endpoints {
            groups.entry(type_id).or_default().push((
                u64::from(rel_id),
                u64::from(from),
                u64::from(to),
            ));
        }
        let mut n = 0;
        for (type_id, mut rows) in sorted_groups(groups) {
            rows.sort_unstable();
            let ids: Vec<u64> = rows.iter().map(|r| r.0).collect();
            let src: Vec<u64> = rows.iter().map(|r| r.1).collect();
            let dst: Vec<u64> = rows.iter().map(|r| r.2).collect();
            v3::write_delete_edge(buf, &ids, type_id as i32, &src, &dst);
            n += 1;
        }
        n
    }
}

/// Groups in a deterministic order.
///
/// A `HashMap` iterates arbitrarily, so without this the same query could emit
/// its records in a different order on two runs — which would defeat the
/// byte-for-byte comparison the cross-engine harness rests on.
fn sorted_groups<K: Ord, V>(groups: FxHashMap<K, V>) -> Vec<(K, V)> {
    let mut out: Vec<(K, V)> = groups.into_iter().collect();
    out.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    out
}

/// The `count x n` row-major values for one shape.
///
/// An entity missing one of the shape's attributes gets `T_NULL` in that slot —
/// unambiguous, because FalkorDB never stores a null property value.
fn gather_rows(
    ids: &[u64],
    attr_ids: &[u16],
    attrs: &FxHashMap<u64, Vec<(u16, Value)>>,
) -> Vec<Value> {
    let mut rows = Vec::with_capacity(ids.len() * attr_ids.len());
    for id in ids {
        let entity = attrs.get(id);
        for attr_id in attr_ids {
            let value = entity
                .and_then(|a| a.iter().find(|(aid, _)| aid == attr_id))
                .map_or(Value::Null, |(_, v)| v.clone());
            rows.push(value);
        }
    }
    rows
}

/// `UPDATE_NODE` / `UPDATE_EDGE`, grouped by attribute shape.
fn emit_updates_v3(
    buf: &mut Vec<u8>,
    attrs: &FxHashMap<u64, Vec<(u16, Value)>>,
    entity: crate::runtime::effects_v3::GraphEntity,
) -> u64 {
    use crate::runtime::effects_v3 as v3;

    let mut groups: FxHashMap<Vec<u16>, Vec<u64>> = FxHashMap::default();
    for (id, pairs) in attrs {
        let shape: Vec<u16> = pairs.iter().map(|(aid, _)| *aid).collect();
        groups.entry(shape).or_default().push(*id);
    }
    let mut n = 0;
    for (attr_ids, mut ids) in sorted_groups(groups) {
        ids.sort_unstable();
        let rows = gather_rows(&ids, &attr_ids, attrs);
        v3::write_update(buf, entity, &ids, &attr_ids, &rows);
        n += 1;
    }
    n
}

/// `SET_LABELS` / `REMOVE_LABELS`, grouped by label set.
///
/// `skip` excludes nodes whose labels already travel inside their
/// `CREATE_NODE` record.
fn emit_labels_v3(
    buf: &mut Vec<u8>,
    labels: &FxHashMap<u64, Vec<u64>>,
    skip: Option<&RoaringTreemap>,
    add: bool,
) -> u64 {
    use crate::runtime::effects_v3 as v3;

    let mut groups: FxHashMap<Vec<i32>, Vec<u64>> = FxHashMap::default();
    for (&id, label_ids) in labels {
        if skip.is_some_and(|s| s.contains(id)) {
            continue;
        }
        let mut shape: Vec<i32> = label_ids.iter().map(|&v| v as i32).collect();
        shape.sort_unstable();
        shape.dedup();
        groups.entry(shape).or_default().push(id);
    }
    let mut n = 0;
    for (label_ids, mut ids) in sorted_groups(groups) {
        ids.sort_unstable();
        v3::write_labels(buf, add, &ids, &label_ids);
        n += 1;
    }
    n
}

// ── Effects buffer constants and helpers ──

pub const EFFECTS_VERSION: u8 = 2;

pub const EFFECT_UPDATE_NODE: u8 = 1;
pub const EFFECT_UPDATE_EDGE: u8 = 2;
pub const EFFECT_CREATE_NODE: u8 = 3;
pub const EFFECT_CREATE_EDGE: u8 = 4;
pub const EFFECT_DELETE_NODE: u8 = 5;
pub const EFFECT_DELETE_EDGE: u8 = 6;
pub const EFFECT_SET_LABELS: u8 = 7;
pub const EFFECT_REMOVE_LABELS: u8 = 8;
pub const EFFECT_ADD_SCHEMA: u8 = 9;
pub const EFFECT_ADD_ATTRIBUTE: u8 = 10;
pub const EFFECT_CREATE_INDEX: u8 = 11;
pub const EFFECT_DROP_INDEX: u8 = 12;

// Schema type tags (used in EFFECT_ADD_SCHEMA)
pub const SCHEMA_NODE_LABEL: u8 = 0;
pub const SCHEMA_REL_TYPE: u8 = 1;

// Attribute type tags (used in EFFECT_ADD_ATTRIBUTE)
pub const ATTR_NODE: u8 = 0;
pub const ATTR_REL: u8 = 1;

// Value type tags for effect serialization
const VALUE_NULL: u8 = 0;
const VALUE_BOOL: u8 = 1;
const VALUE_INT: u8 = 2;
const VALUE_FLOAT: u8 = 3;
const VALUE_STRING: u8 = 4;
const VALUE_LIST: u8 = 5;
const VALUE_POINT: u8 = 6;
const VALUE_VECF32: u8 = 7;
const VALUE_DATETIME: u8 = 8;
const VALUE_DATE: u8 = 9;
const VALUE_TIME: u8 = 10;
const VALUE_DURATION: u8 = 11;
const VALUE_INTERN_STRING: u8 = 12;

pub fn write_u16(
    buf: &mut Vec<u8>,
    v: u16,
) {
    buf.extend_from_slice(&v.to_le_bytes());
}

pub fn write_string(
    buf: &mut Vec<u8>,
    s: &str,
) {
    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

fn write_value(
    buf: &mut Vec<u8>,
    value: &Value,
) {
    match value {
        Value::Null => buf.push(VALUE_NULL),
        Value::Bool(b) => {
            buf.push(VALUE_BOOL);
            buf.push(u8::from(*b));
        }
        Value::Int(i) => {
            buf.push(VALUE_INT);
            buf.extend_from_slice(&i.to_le_bytes());
        }
        Value::Float(f) => {
            buf.push(VALUE_FLOAT);
            buf.extend_from_slice(&f.to_le_bytes());
        }
        Value::String(s) => {
            if crate::runtime::string_pool::global().is_interned(s) {
                buf.push(VALUE_INTERN_STRING);
            } else {
                buf.push(VALUE_STRING);
            }
            write_string(buf, s);
        }
        Value::List(items) => {
            buf.push(VALUE_LIST);
            buf.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for item in items.iter() {
                write_value(buf, item);
            }
        }
        Value::Point(p) => {
            buf.push(VALUE_POINT);
            buf.extend_from_slice(&(p.latitude as f64).to_le_bytes());
            buf.extend_from_slice(&(p.longitude as f64).to_le_bytes());
        }
        Value::VecF32(v) => {
            buf.push(VALUE_VECF32);
            buf.extend_from_slice(&(v.len() as u64).to_le_bytes());
            for f in v.iter() {
                buf.extend_from_slice(&f.to_le_bytes());
            }
        }
        Value::Datetime(ts) => {
            buf.push(VALUE_DATETIME);
            buf.extend_from_slice(&ts.to_le_bytes());
        }
        Value::Date(ts) => {
            buf.push(VALUE_DATE);
            buf.extend_from_slice(&ts.to_le_bytes());
        }
        Value::Time(ts) => {
            buf.push(VALUE_TIME);
            buf.extend_from_slice(&ts.to_le_bytes());
        }
        Value::Duration(dur) => {
            buf.push(VALUE_DURATION);
            buf.extend_from_slice(&dur.to_le_bytes());
        }
        _ => {
            debug_assert!(false, "Unsupported value type in effects buffer: {value:?}");
            buf.push(VALUE_NULL); // Fallback for unsupported types
        }
    }
}

pub fn read_string(
    buf: &[u8],
    offset: &mut usize,
) -> Result<Arc<String>, String> {
    if *offset + 8 > buf.len() {
        return Err("effects buffer truncated".to_string());
    }
    let len = u64::from_le_bytes(buf[*offset..*offset + 8].try_into().unwrap()) as usize;
    *offset += 8;
    if *offset + len > buf.len() {
        return Err("effects buffer truncated".to_string());
    }
    let s = std::str::from_utf8(&buf[*offset..*offset + len])
        .map_err(|e| format!("invalid utf8 in effects buffer: {e}"))?;
    *offset += len;
    Ok(Arc::new(s.to_string()))
}

pub fn read_u16(
    buf: &[u8],
    offset: &mut usize,
) -> Result<u16, String> {
    if *offset + 2 > buf.len() {
        return Err("effects buffer truncated".to_string());
    }
    let v = u16::from_le_bytes(buf[*offset..*offset + 2].try_into().unwrap());
    *offset += 2;
    Ok(v)
}

pub fn read_u64(
    buf: &[u8],
    offset: &mut usize,
) -> Result<u64, String> {
    if *offset + 8 > buf.len() {
        return Err("effects buffer truncated".to_string());
    }
    let v = u64::from_le_bytes(buf[*offset..*offset + 8].try_into().unwrap());
    *offset += 8;
    Ok(v)
}

pub fn read_value(
    buf: &[u8],
    offset: &mut usize,
) -> Result<Value, String> {
    if *offset >= buf.len() {
        return Err("effects buffer truncated".to_string());
    }
    let tag = buf[*offset];
    *offset += 1;
    match tag {
        VALUE_NULL => Ok(Value::Null),
        VALUE_BOOL => {
            if *offset >= buf.len() {
                return Err("effects buffer truncated".to_string());
            }
            let b = buf[*offset] != 0;
            *offset += 1;
            Ok(Value::Bool(b))
        }
        VALUE_INT => {
            let v = i64::from_le_bytes(
                buf.get(*offset..*offset + 8)
                    .ok_or("truncated")?
                    .try_into()
                    .unwrap(),
            );
            *offset += 8;
            Ok(Value::Int(v))
        }
        VALUE_FLOAT => {
            let v = f64::from_le_bytes(
                buf.get(*offset..*offset + 8)
                    .ok_or("truncated")?
                    .try_into()
                    .unwrap(),
            );
            *offset += 8;
            Ok(Value::Float(v))
        }
        VALUE_STRING => {
            let s = read_string(buf, offset)?;
            Ok(Value::String(s))
        }
        VALUE_INTERN_STRING => {
            let s = read_string(buf, offset)?;
            Ok(Value::String(
                crate::runtime::string_pool::global().intern(s),
            ))
        }
        VALUE_LIST => {
            let len = read_u64(buf, offset)? as usize;
            let mut items = thin_vec::ThinVec::with_capacity(len);
            for _ in 0..len {
                items.push(read_value(buf, offset)?);
            }
            Ok(Value::List(Arc::new(items)))
        }
        VALUE_POINT => {
            let lat = f64::from_le_bytes(
                buf.get(*offset..*offset + 8)
                    .ok_or("truncated")?
                    .try_into()
                    .unwrap(),
            );
            *offset += 8;
            let lon = f64::from_le_bytes(
                buf.get(*offset..*offset + 8)
                    .ok_or("truncated")?
                    .try_into()
                    .unwrap(),
            );
            *offset += 8;
            Ok(Value::Point(crate::runtime::value::Point {
                latitude: lat as f32,
                longitude: lon as f32,
            }))
        }
        VALUE_VECF32 => {
            let len = read_u64(buf, offset)? as usize;
            let mut v = Vec::with_capacity(len);
            for _ in 0..len {
                let f = f32::from_le_bytes(
                    buf.get(*offset..*offset + 4)
                        .ok_or("truncated")?
                        .try_into()
                        .unwrap(),
                );
                *offset += 4;
                v.push(f);
            }
            Ok(Value::VecF32(Arc::new(v.into())))
        }
        VALUE_DATETIME => {
            let ts = i64::from_le_bytes(
                buf.get(*offset..*offset + 8)
                    .ok_or("truncated")?
                    .try_into()
                    .unwrap(),
            );
            *offset += 8;
            Ok(Value::Datetime(ts))
        }
        VALUE_DATE => {
            let ts = i64::from_le_bytes(
                buf.get(*offset..*offset + 8)
                    .ok_or("truncated")?
                    .try_into()
                    .unwrap(),
            );
            *offset += 8;
            Ok(Value::Date(ts))
        }
        VALUE_TIME => {
            let ts = i64::from_le_bytes(
                buf.get(*offset..*offset + 8)
                    .ok_or("truncated")?
                    .try_into()
                    .unwrap(),
            );
            *offset += 8;
            Ok(Value::Time(ts))
        }
        VALUE_DURATION => {
            let dur = i64::from_le_bytes(
                buf.get(*offset..*offset + 8)
                    .ok_or("truncated")?
                    .try_into()
                    .unwrap(),
            );
            *offset += 8;
            Ok(Value::Duration(dur))
        }
        _ => Err(format!("unknown value tag in effects buffer: {tag}")),
    }
}

#[cfg(test)]
mod v3_tests {
    use super::*;
    use crate::graph::graphblas::test_init::ensure_init;
    use crate::runtime::effects_v3::{GraphEntity, Record, read_buffer};

    fn graph() -> AtomicRefCell<Graph> {
        // GrB_init is process-wide and may only run once.
        ensure_init();
        AtomicRefCell::new(Graph::new(64, 64, 0, 0, "t"))
    }

    /// Register `n` node attributes so the ids the tests use resolve.
    fn with_attrs(
        g: &AtomicRefCell<Graph>,
        names: &[&str],
    ) {
        let mut graph = g.borrow_mut();
        for name in names {
            graph.add_node_attribute_name(name);
        }
    }

    fn build(
        p: &Pending,
        g: &AtomicRefCell<Graph>,
    ) -> Vec<Record> {
        let mut buf = Vec::new();
        p.build_effects_buffer_v3(g, &mut buf);
        read_buffer(&buf).expect("v3 buffer must decode")
    }

    #[test]
    fn nodes_of_one_shape_become_one_record() {
        let g = graph();
        with_attrs(&g, &["a"]);
        let mut p = Pending::default();
        p.set_schema_baseline(&g);
        for id in 0..100u64 {
            p.created_nodes.insert(id);
            p.set_labels.insert(id, vec![0]);
            p.new_nodes_attrs
                .insert(id, vec![(0, Value::Int(id as i64))]);
        }

        let records = build(&p, &g);
        assert_eq!(records.len(), 1, "100 nodes, one shape, one record");
        let Record::CreateNode {
            ids,
            labels,
            attr_ids,
            rows,
        } = &records[0]
        else {
            panic!("wrong record: {:?}", records[0]);
        };
        assert_eq!(ids.len(), 100);
        assert_eq!(labels, &vec![0]);
        assert_eq!(attr_ids, &vec![0]);
        assert_eq!(rows.len(), 100);
        // Row k belongs to the k-th smallest id.
        assert_eq!(rows[7], Value::Int(7));
    }

    #[test]
    fn label_order_does_not_split_a_shape() {
        // set_labels keeps caller order, so [0,1] and [1,0] arrive differently
        // for the same set. They must land in one record, sorted.
        let g = graph();
        let mut p = Pending::default();
        p.set_schema_baseline(&g);
        p.created_nodes.insert(1);
        p.set_labels.insert(1, vec![0, 1]);
        p.created_nodes.insert(2);
        p.set_labels.insert(2, vec![1, 0]);

        let records = build(&p, &g);
        assert_eq!(records.len(), 1, "same label set, so one record");
        let Record::CreateNode { ids, labels, .. } = &records[0] else {
            panic!("wrong record");
        };
        assert_eq!(ids, &vec![1, 2]);
        assert_eq!(
            labels,
            &vec![0, 1],
            "emitted ascending, so bytes are canonical"
        );
    }

    #[test]
    fn differing_shapes_split_into_separate_records() {
        let g = graph();
        with_attrs(&g, &["a", "b"]);
        let mut p = Pending::default();
        p.set_schema_baseline(&g);
        p.created_nodes.insert(1);
        p.new_nodes_attrs.insert(1, vec![(0, Value::Int(1))]);
        p.created_nodes.insert(2);
        p.new_nodes_attrs.insert(2, vec![(1, Value::Int(2))]);

        let records = build(&p, &g);
        assert_eq!(
            records.len(),
            2,
            "different attribute ids are different shapes"
        );
    }

    #[test]
    fn a_created_node_carries_its_labels_and_emits_no_set_labels() {
        let g = graph();
        let mut p = Pending::default();
        p.set_schema_baseline(&g);
        p.created_nodes.insert(1);
        p.set_labels.insert(1, vec![3]);
        // A node that already existed does need a SET_LABELS record.
        p.set_labels.insert(99, vec![3]);

        let records = build(&p, &g);
        assert_eq!(records.len(), 2);
        assert!(matches!(records[0], Record::CreateNode { .. }));
        let Record::Labels { add, ids, labels } = &records[1] else {
            panic!("expected SET_LABELS for the pre-existing node");
        };
        assert!(add);
        assert_eq!(ids, &vec![99], "the created node is not repeated here");
        assert_eq!(labels, &vec![3]);
    }

    #[test]
    fn updates_group_by_attribute_shape() {
        let g = graph();
        with_attrs(&g, &["a", "b"]);
        let mut p = Pending::default();
        p.set_schema_baseline(&g);
        p.existing_nodes_attrs.insert(1, vec![(0, Value::Int(1))]);
        p.existing_nodes_attrs.insert(2, vec![(0, Value::Int(2))]);
        p.existing_nodes_attrs
            .insert(3, vec![(0, Value::Int(3)), (1, Value::Int(4))]);

        let records = build(&p, &g);
        assert_eq!(records.len(), 2, "two shapes");
        let Record::Update { entity, ids, .. } = &records[0] else {
            panic!("wrong record");
        };
        assert_eq!(*entity, GraphEntity::Node);
        assert_eq!(ids, &vec![1, 2]);
    }

    #[test]
    fn schema_records_precede_the_records_that_use_them() {
        // Normative ordering: an id on the wire means nothing until the replica
        // has agreed on the numbering that produced it.
        let g = graph();
        {
            let mut graph = g.borrow_mut();
            graph.get_label_id_mut("L");
            graph.add_node_attribute_name("a");
        }
        // Default counts are already zero, which is what "the schema is new"
        // means: everything the graph holds gets announced.
        let mut p = Pending::default();
        p.created_nodes.insert(1);
        p.set_labels.insert(1, vec![0]);
        p.new_nodes_attrs.insert(1, vec![(0, Value::Int(1))]);

        let records = build(&p, &g);
        assert_eq!(records.len(), 3);
        assert_eq!(
            records[0],
            Record::AddSchema {
                schema_type: 0,
                id: 0,
                name: "L".into()
            }
        );
        assert_eq!(
            records[1],
            Record::AddAttribute {
                id: 0,
                name: "a".into()
            }
        );
        assert!(matches!(records[2], Record::CreateNode { .. }));
    }

    #[test]
    fn grouping_is_deterministic() {
        // Two Pendings built in different insertion orders must produce
        // byte-identical buffers, or the cross-engine comparison is worthless.
        let g = graph();
        with_attrs(&g, &["a", "b"]);
        let mut forward = Pending::default();
        let mut backward = Pending::default();
        forward.set_schema_baseline(&g);
        backward.set_schema_baseline(&g);
        for id in 0..50u64 {
            forward.created_nodes.insert(id);
            forward
                .new_nodes_attrs
                .insert(id, vec![((id % 2) as u16, Value::Int(id as i64))]);
        }
        for id in (0..50u64).rev() {
            backward.created_nodes.insert(id);
            backward
                .new_nodes_attrs
                .insert(id, vec![((id % 2) as u16, Value::Int(id as i64))]);
        }

        let mut a = Vec::new();
        forward.build_effects_buffer_v3(&g, &mut a);
        let mut b = Vec::new();
        backward.build_effects_buffer_v3(&g, &mut b);
        assert_eq!(a, b, "insertion order must not reach the wire");
    }

    #[test]
    fn a_new_attribute_is_announced_once() {
        // #2459 unified the node and relationship attribute dictionaries, so
        // the two getters return the same store. v3's ADD_ATTRIBUTE carries no
        // discriminator, so announcing per entity kind would send the same
        // attribute twice under the same id.
        let g = graph();
        {
            let mut graph = g.borrow_mut();
            graph.add_node_attribute_name("a");
            graph.add_node_attribute_name("b");
        }
        // Default counts are zero, so both attributes are new.
        let p = Pending::default();

        let records = build(&p, &g);
        assert_eq!(
            records,
            vec![
                Record::AddAttribute {
                    id: 0,
                    name: "a".into()
                },
                Record::AddAttribute {
                    id: 1,
                    name: "b".into()
                },
            ]
        );
    }

    #[test]
    fn deleted_edges_group_by_relationship_type() {
        // The type has to be captured at deletion time — the edge is gone by
        // the time effects are built, and index_remove_edge_docs only knows
        // about types that carry an index.
        let g = graph();
        let mut p = Pending::default();
        p.set_schema_baseline(&g);
        p.deleted_endpoints = vec![
            (1.into(), 0, 10.into(), 11.into()),
            (2.into(), 1, 12.into(), 13.into()),
            (3.into(), 0, 14.into(), 15.into()),
        ];

        let records = build(&p, &g);
        assert_eq!(records.len(), 2, "two types, two records");
        let Record::DeleteEdge {
            ids,
            relation_id,
            src,
            dst,
        } = &records[0]
        else {
            panic!("wrong record: {:?}", records[0]);
        };
        assert_eq!(*relation_id, 0);
        assert_eq!(ids, &vec![1, 3]);
        assert_eq!(src, &vec![10, 14]);
        assert_eq!(dst, &vec![11, 15]);
    }

    #[test]
    fn v3_is_smaller_than_v2_for_the_motivating_query() {
        let g = graph();
        with_attrs(&g, &["v"]);
        let mut p = Pending::default();
        p.set_schema_baseline(&g);
        for id in 0..10_000u64 {
            p.created_nodes.insert(id);
            p.set_labels.insert(id, vec![0]);
            p.new_nodes_attrs
                .insert(id, vec![(0, Value::Int(id as i64))]);
        }

        let mut v2 = Vec::new();
        p.build_effects_buffer(&g, &mut v2);
        let mut v3 = Vec::new();
        p.build_effects_buffer_v3(&g, &mut v3);
        println!("\n  10,000 nodes: v2 {} B -> v3 {} B", v2.len(), v3.len());
        assert!(v3.len() * 2 < v2.len(), "v2 {} v3 {}", v2.len(), v3.len());
        assert_eq!(read_buffer(&v3).unwrap().len(), 1);
    }
}
