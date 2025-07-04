// Copyright 2012 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS-IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

// Author: ericv@google.com (Eric Veach)

#ifndef S2_MUTABLE_S2SHAPE_INDEX_H_
#define S2_MUTABLE_S2SHAPE_INDEX_H_

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/macros.h"
// #include "absl/base/thread_annotations.h"
#include "absl/container/btree_map.h"
#include "absl/log/absl_check.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"

#include "s2/_fp_contract_off.h"  // IWYU pragma: keep
#include "s2/base/commandlineflags.h"
#include "s2/base/commandlineflags_declare.h"
#include "s2/base/spinlock.h"
#include "s2/r1interval.h"
#include "s2/r2rect.h"
#include "s2/s2cell_id.h"
#include "s2/s2memory_tracker.h"
#include "s2/s2point.h"
#include "s2/s2pointutil.h"
#include "s2/s2shape.h"
#include "s2/s2shape_index.h"
#include "s2/s2shapeutil_shape_edge_id.h"
#include "s2/util/coding/coder.h"

class S2PaddedCell;

namespace s2internal {
// Hack to expose bytes_used.
template <typename Key, typename Value>
class BTreeMap : public absl::btree_map<Key, Value> {
 public:
  size_t bytes_used() const { return this->tree_.bytes_used(); }
};
}  // namespace s2internal

// MutableS2ShapeIndex is a class for in-memory indexing of polygonal geometry.
// The objects in the index are known as "shapes", and may consist of points,
// polylines, and/or polygons, possibly overlapping.  The index makes it very
// fast to answer queries such as finding nearby shapes, measuring distances,
// testing for intersection and containment, etc.  It is one of several
// implementations of the S2ShapeIndex interface (see EncodedS2ShapeIndex).
//
// MutableS2ShapeIndex allows not only building an index, but also updating it
// incrementally by adding or removing shapes (hence its name).  It is designed
// to be compact; usually the index is smaller than the underlying geometry.
// It is capable of indexing up to hundreds of millions of edges.  The index is
// also fast to construct.  The index size and construction time are guaranteed
// to be linear in the number of input edges.
//
// There are a number of built-in classes that work with S2ShapeIndex objects.
// Generally these classes accept any collection of geometry that can be
// represented by an S2ShapeIndex, i.e. any combination of points, polylines,
// and polygons.  Such classes include:
//
// - S2ContainsPointQuery: returns the shape(s) that contain a given point.
//
// - S2ClosestEdgeQuery: returns the closest edge(s) to a given point, edge,
//                       S2CellId, or S2ShapeIndex.
//
// - S2CrossingEdgeQuery: returns the edge(s) that cross a given edge.
//
// - S2BooleanOperation: computes boolean operations such as union,
//                       and boolean predicates such as containment.
//
// - S2ShapeIndexRegion: can be used together with S2RegionCoverer to
//                       approximate geometry as a set of S2CellIds.
//
// - S2ShapeIndexBufferedRegion: computes approximations that have been
//                               expanded by a given radius.
//
// Here is an example showing how to build an index for a set of polygons, and
// then then determine which polygon(s) contain each of a set of query points:
//
//   void TestContainment(const vector<S2Point>& points,
//                        const vector<S2Polygon*>& polygons) {
//     MutableS2ShapeIndex index;
//     for (auto polygon : polygons) {
//       index.Add(std::make_unique<S2Polygon::Shape>(polygon));
//     }
//     auto query = MakeS2ContainsPointQuery(&index);
//     for (const auto& point : points) {
//       for (int shape_id : query.GetContainingShapeIds(point)) {
//         S2Polygon* polygon = polygons[shape_id];
//         ... do something with (point, polygon) ...
//       }
//     }
//   }
//
// This example uses S2Polygon::Shape, which is one example of an S2Shape
// object.  S2Polyline and S2Loop also have nested Shape classes, and there are
// additional S2Shape types defined in *_shape.h.
//
// Internally, MutableS2ShapeIndex is essentially a map from S2CellIds to the
// set of shapes that intersect each S2CellId.  It is adaptively refined to
// ensure that no cell contains more than a small number of edges.
//
// For efficiency, updates are batched together and applied lazily on the
// first subsequent query.  Locking is used to ensure that MutableS2ShapeIndex
// has the same thread-safety properties as "vector": const methods are
// thread-safe, while non-const methods are not thread-safe.  This means that
// if one thread updates the index, you must ensure that no other thread is
// reading or updating the index at the same time.
//
// MutableS2ShapeIndex has an Encode() method that allows the index to be
// serialized.  An encoded S2ShapeIndex can be decoded either into its
// original form (MutableS2ShapeIndex) or into an EncodedS2ShapeIndex.  The
// key property of EncodedS2ShapeIndex is that it can be constructed
// instantaneously, since the index is kept in its original encoded form.
// Data is decoded only when an operation needs it.  For example, to determine
// which shapes(s) contain a given query point only requires decoding the data
// in the S2ShapeIndexCell that contains that point.
class MutableS2ShapeIndex final : public S2ShapeIndex {
 private:
  using CellMap =
      s2internal::BTreeMap<S2CellId, std::unique_ptr<S2ShapeIndexCell>>;

 public:
  // The amount by which cells are "padded" to compensate for numerical errors
  // when clipping line segments to cell boundaries.
  static const double kCellPadding;

  // Options that affect construction of the MutableS2ShapeIndex.
  class Options {
   public:
    Options();

    // The maximum number of edges per cell.  If a cell has more than this
    // many edges that are not considered "long" relative to the cell size,
    // then it is subdivided.  (Whether an edge is considered "long" is
    // controlled by --s2shape_index_cell_size_to_long_edge_ratio flag.)
    //
    // Values between 10 and 50 represent a reasonable balance between memory
    // usage, construction time, and query time.  Small values make queries
    // faster, while large values make construction faster and use less memory.
    // Values higher than 50 do not save significant additional memory, and
    // query times can increase substantially, especially for algorithms that
    // visit all pairs of potentially intersecting edges (such as polygon
    // validation), since this is quadratic in the number of edges per cell.
    //
    // Note that the *average* number of edges per cell is generally slightly
    // less than half of the maximum value defined here.
    //
    // Defaults to value given by --s2shape_index_default_max_edges_per_cell.
    int max_edges_per_cell() const { return max_edges_per_cell_; }
    void set_max_edges_per_cell(int max_edges_per_cell);

   private:
    int max_edges_per_cell_;
  };

  // Creates a MutableS2ShapeIndex that uses the default option settings.
  // Option values may be changed by calling Init().
  MutableS2ShapeIndex();

  // Create a MutableS2ShapeIndex with the given options.
  explicit MutableS2ShapeIndex(const Options& options);

  ~MutableS2ShapeIndex() override;

  MutableS2ShapeIndex(MutableS2ShapeIndex&&) noexcept;
  MutableS2ShapeIndex& operator=(MutableS2ShapeIndex&&) noexcept;

  // Initialize a MutableS2ShapeIndex with the given options.  This method may
  // only be called when the index is empty (i.e. newly created or Clear() has
  // just been called).  May be called before or after set_memory_tracker().
  void Init(const Options& options);

  const Options& options() const { return options_; }

  // Specifies that memory usage should be tracked and/or limited by the given
  // S2MemoryTracker.  For example:
  //
  //   S2MemoryTracker tracker;
  //   tracker.set_limit(500 << 20);  // 500 MB memory limit
  //   MutableS2ShapeIndex index;
  //   index.set_memory_tracker(&tracker);
  //
  // If the memory limit is exceeded, an appropriate status is returned in
  // memory_tracker()->error() and any partially built index is discarded
  // (equivalent to calling Minimize()).
  //
  // This method may be called multiple times in order to switch from one
  // memory tracker to another or stop memory tracking altogether (by passing
  // nullptr) in which case the memory usage due to this index is subtracted.
  //
  // REQUIRES: The lifetime of "tracker" must exceed the lifetime of the index
  //           unless set_memory_tracker(nullptr) is called to stop memory
  //           tracking before the index destructor is called.
  //
  //           This implies that the S2MemoryTracker must be declared *before*
  //           the MutableS2ShapeIndex in the example above.
  //
  // CAVEATS:
  //
  //  - This method is not const and is therefore not thread-safe.
  //
  //  - Does not track memory used by the S2Shapes in the index.
  //
  //  - While the index representation itself is tracked very accurately,
  //    the temporary data needed for index construction is tracked using
  //    heuristics and may be underestimated or overestimated.
  //
  //  - Temporary memory usage is typically 10x larger than the final index
  //    size, however it can be reduced by specifying a suitable value for
  //    FLAGS_s2shape_index_tmp_memory_budget (the default is 100 MB).  If
  //    more temporary memory than this is needed during construction, index
  //    updates will be split into multiple batches in order to keep the
  //    estimated temporary memory usage below this limit.
  //
  //  - S2MemoryTracker::limit() has no effect on how much temporary memory
  //    MutableS2ShapeIndex will attempt to use during index construction; it
  //    simply causes an error to be returned when the limit would otherwise
  //    be exceeded.  If you set a memory limit smaller than 100MB and want to
  //    reduce memory usage rather than simply generating an error then you
  //    should also set FLAGS_s2shape_index_tmp_memory_budget appropriately.
  void set_memory_tracker(S2MemoryTracker* tracker);
  S2MemoryTracker* memory_tracker() const { return mem_tracker_.tracker(); }

  // The number of distinct shape ids that have been assigned.  This equals
  // the number of shapes in the index provided that no shapes have ever been
  // removed.  (Shape ids are not reused.)
  int num_shape_ids() const override {
    return static_cast<int>(shapes_.size());
  }

  // Returns a pointer to the shape with the given id, or nullptr if the shape
  // has been removed from the index.
  const S2Shape* shape(int id) const override { return shapes_[id].get(); }

  // Minimizes memory usage by requesting that any data structures that can be
  // rebuilt should be discarded.  This method invalidates all iterators.
  //
  // Like all non-const methods, this method is not thread-safe.
  void Minimize() override;

  // Appends an encoded representation of the S2ShapeIndex to "encoder".
  //
  // This method does not encode the S2Shapes in the index; it is the client's
  // responsibility to encode them separately.  For example:
  //
  //   s2shapeutil::CompactEncodeTaggedShapes(index, encoder);
  //   index.Encode(encoder);
  //
  // The encoded size is typically much smaller than the in-memory size.
  // Here are a few examples:
  //
  //  Number of edges     In-memory space used     Encoded size  (%)
  //  --------------------------------------------------------------
  //                8                      192                8   4%
  //              768                   18,264            2,021  11%
  //        3,784,212               80,978,992       17,039,020  21%
  //
  // The encoded form also has the advantage of being a contiguous block of
  // memory.
  //
  // REQUIRES: "encoder" uses the default constructor, so that its buffer
  //           can be enlarged as necessary by calling Ensure(int).
  void Encode(Encoder* encoder) const override;

  // Decodes an S2ShapeIndex, returning true on success.
  //
  // This method does not decode the S2Shape objects in the index; this is
  // the responsibility of the client-provided function "shape_factory"
  // (see s2shapeutil_coding.h).  Example usage:
  //
  //   index.Init(decoder, s2shapeutil::LazyDecodeShapeFactory(decoder));
  //
  // Note that the S2Shape vector must be encoded *before* the S2ShapeIndex in
  // this example.
  bool Init(Decoder* decoder, const ShapeFactory& shape_factory);

  class Iterator final : public IteratorBase {
   public:
    // Default constructor; must be followed by a call to Init().
    Iterator() = default;

    // Constructs an iterator positioned as specified.  By default iterators
    // are unpositioned, since this avoids an extra seek in this situation
    // where one of the seek methods (such as Locate) is immediately called.
    //
    // If you want to position the iterator at the beginning, e.g. in order to
    // loop through the entire index, do this instead:
    //
    //   for (MutableS2ShapeIndex::Iterator it(&index, S2ShapeIndex::BEGIN);
    //        !it.done(); it.Next()) { ... }
    explicit Iterator(const MutableS2ShapeIndex* index,
                      InitialPosition pos = UNPOSITIONED);

    // Initializes an iterator for the given MutableS2ShapeIndex.  This method
    // may also be called in order to restore an iterator to a valid state
    // after the underlying index has been updated (although it is usually
    // easier just to declare a new iterator whenever required, since iterator
    // construction is cheap).
    void Init(const MutableS2ShapeIndex* index,
              InitialPosition pos = UNPOSITIONED);

    // Initialize an iterator for the given MutableS2ShapeIndex without
    // applying any pending updates.  This can be used to observe the actual
    // current state of the index without modifying it in any way.
    void InitStale(const MutableS2ShapeIndex* index,
                   InitialPosition pos = UNPOSITIONED);

    S2CellId id() const override {
      S2CellId id = S2CellId::Sentinel();
      if (!done()) {
        id = iter_->first;
      }
      return id;
    }

    const S2ShapeIndexCell& cell() const override {
      ABSL_DCHECK(!done());
      return *iter_->second;
    }

    bool done() const override { return iter_ == end_; }

    // S2CellIterator API:
    void Begin() override;
    void Finish() override;
    void Next() override;
    bool Prev() override;
    void Seek(S2CellId target) override;

    bool Locate(const S2Point& target) override {
      return LocateImpl(*this, target);
    }

    S2CellRelation Locate(S2CellId target) override {
      return LocateImpl(*this, target);
    }

    std::unique_ptr<IteratorBase> Clone() const override {
      return std::make_unique<Iterator>(*this);
    }

   private:
    const MutableS2ShapeIndex* index_ = nullptr;
    CellMap::const_iterator iter_, end_;
  };

  // Takes ownership of the given shape and adds it to the index.  Assigns a
  // unique id to the shape for this index and returns it.  Shape ids are
  // assigned sequentially starting from 0 in the order shapes are added.
  // Invalidates all iterators and their associated data.
  //
  // Note that this method is not affected by S2MemoryTracker, i.e. shapes can
  // continue to be added even once the specified limit has been reached.
  int Add(std::unique_ptr<S2Shape> shape);

  // Removes the given shape from the index and return ownership to the caller.
  // Invalidates all iterators and their associated data.
  std::unique_ptr<S2Shape> Release(int shape_id);

  // Resets the index to its original state and returns ownership of all
  // shapes to the caller.  This method is much more efficient than removing
  // all shapes one at a time.
  std::vector<std::unique_ptr<S2Shape>> ReleaseAll();

  // Resets the index to its original state and deletes all shapes.  Any
  // options specified via Init() are preserved.
  void Clear();

  // Returns the number of bytes currently occupied by the index (including any
  // unused space at the end of vectors, etc). It has the same thread safety
  // as the other "const" methods (see introduction).
  size_t SpaceUsed() const override;

  // Calls to Add() and Release() are normally queued and processed on the
  // first subsequent query (in a thread-safe way).  Building the index lazily
  // in this way has several advantages, the most important of which is that
  // sometimes there *is* no subsequent query and the index doesn't need to be
  // built at all.
  //
  // In contrast, ForceBuild() causes any pending updates to be applied
  // immediately.  It is thread-safe and may be called simultaneously with
  // other "const" methods (see notes on thread safety above).  Similarly this
  // method is "const" since it does not modify the visible index contents.
  //
  // ForceBuild() should not normally be called since it prevents lazy index
  // construction (which is usually benficial).  Some reasons to use it
  // include:
  //
  //  - To exclude the cost of building the index from benchmark results.
  //  - To ensure that the first subsequent query is as fast as possible.
  //  - To ensure that the index can be built successfully without exceeding a
  //    specified S2MemoryTracker limit (see the constructor for details).
  //
  // Note that this method is thread-safe.
  void ForceBuild() const;

  // Returns true if there are no pending updates that need to be applied.
  // This can be useful to avoid building the index unnecessarily, or for
  // choosing between two different algorithms depending on whether the index
  // is available.
  //
  // The returned index status may be slightly out of date if the index was
  // built in a different thread.  This is fine for the intended use (as an
  // efficiency hint), but it should not be used by internal methods  (see
  // MaybeApplyUpdates).
  bool is_fresh() const;

 protected:
  std::unique_ptr<IteratorBase> NewIterator(InitialPosition pos) const override;

 private:
  friend class EncodedS2ShapeIndex;
  friend class Iterator;
  friend class MutableS2ShapeIndexTest;
  friend class S2Stats;

  class BatchGenerator;
  class EdgeAllocator;
  class InteriorTracker;
  struct BatchDescriptor;
  struct ClippedEdge;
  struct FaceEdge;
  struct RemovedShape;

  using ShapeEdgeId = s2shapeutil::ShapeEdgeId;
  using ShapeIdSet = std::vector<int>;

  // When adding a new encoding, be aware that old binaries will not be able
  // to decode it.
  static constexpr unsigned char kCurrentEncodingVersionNumber = 0;

  // Internal methods are documented with their definitions.
  bool is_shape_being_removed(int shape_id) const;
  void MarkIndexStale();
  void MaybeApplyUpdates() const;
  void ApplyUpdatesThreadSafe();
  void ApplyUpdatesInternal();
  std::vector<BatchDescriptor> GetUpdateBatches() const;
  void ReserveSpace(const BatchDescriptor& batch,
                    std::vector<FaceEdge> all_edges[6]);
  void AddShape(const S2Shape* shape, int shape_id, int edges_begin,
                int edges_end, std::vector<FaceEdge> all_edges[6],
                InteriorTracker* tracker) const;
  void RemoveShape(const RemovedShape& removed,
                   std::vector<FaceEdge> all_edges[6],
                   InteriorTracker* tracker) const;
  void FinishPartialShape(int shape_id);
  void AddFaceEdge(FaceEdge* edge, std::vector<FaceEdge> all_edges[6]) const;
  void UpdateFaceEdges(int face, absl::Span<const FaceEdge> face_edges,
                       InteriorTracker* tracker);
  S2CellId ShrinkToFit(const S2PaddedCell& pcell, const R2Rect& bound) const;
  void SkipCellRange(S2CellId begin, S2CellId end, InteriorTracker* tracker,
                     EdgeAllocator* alloc, bool disjoint_from_index);
  void UpdateEdges(const S2PaddedCell& pcell,
                   std::vector<const ClippedEdge*>* edges,
                   InteriorTracker* tracker, EdgeAllocator* alloc,
                   bool disjoint_from_index);
  void AbsorbIndexCell(const S2PaddedCell& pcell,
                       const Iterator& iter,
                       std::vector<const ClippedEdge*>* edges,
                       InteriorTracker* tracker,
                       EdgeAllocator* alloc);
  int GetEdgeMaxLevel(const S2Shape::Edge& edge) const;
  static int CountShapes(const std::vector<const ClippedEdge*>& edges,
                         const ShapeIdSet& cshape_ids);
  bool MakeIndexCell(const S2PaddedCell& pcell,
                     const std::vector<const ClippedEdge*>& edges,
                     InteriorTracker* tracker);
  static void TestAllEdges(const std::vector<const ClippedEdge*>& edges,
                           InteriorTracker* tracker);
  inline static const ClippedEdge* UpdateBound(const ClippedEdge* edge,
                                               int u_end, double u,
                                               int v_end, double v,
                                               EdgeAllocator* alloc);
  static const ClippedEdge* ClipUBound(const ClippedEdge* edge,
                                       int u_end, double u,
                                       EdgeAllocator* alloc);
  static const ClippedEdge* ClipVBound(const ClippedEdge* edge,
                                       int v_end, double v,
                                       EdgeAllocator* alloc);
  static void ClipVAxis(const ClippedEdge* edge, const R1Interval& middle,
                        std::vector<const ClippedEdge*> child_edges[2],
                        EdgeAllocator* alloc);

  // The shapes in the index, accessed by their shape id.  Removed shapes are
  // replaced by nullptr pointers.
  std::vector<std::unique_ptr<S2Shape>> shapes_;

  // A map from S2CellId to the set of clipped shapes that intersect that
  // cell.  The cell ids cover a set of non-overlapping regions on the
  // sphere.  Note that this field is updated lazily (see below).  Const
  // methods *must* call MaybeApplyUpdates() before accessing this field.
  // (The easiest way to achieve this is simply to use an Iterator.)
  CellMap cell_map_;

  // The options supplied for this index.
  Options options_;

  // The id of the first shape that has been queued for addition but not
  // processed yet.
  int pending_additions_begin_ = 0;

  // The representation of an edge that has been queued for removal.
  struct RemovedShape {
    int32_t shape_id;
    bool has_interior;  // Belongs to a shape of dimension 2.
    bool contains_tracker_origin;
    std::vector<S2Shape::Edge> edges;
  };

  // The set of shapes that have been queued for removal but not processed
  // yet.  Note that we need to copy the edge data since the caller is free to
  // destroy the shape once Release() has been called.  This field is present
  // only when there are removed shapes to process (to save memory).
  std::unique_ptr<std::vector<RemovedShape>> pending_removals_;

  // Additions and removals are queued and processed on the first subsequent
  // query.  There are several reasons to do this:
  //
  //  - It is significantly more efficient to process updates in batches.
  //  - Often the index will never be queried, in which case we can save both
  //    the time and memory required to build it.  Examples:
  //     + S2Loops that are created simply to pass to an S2Polygon.  (We don't
  //       need the S2Loop index, because S2Polygon builds its own index.)
  //     + Applications that load a database of geometry and then query only
  //       a small fraction of it.
  //     + Applications that only read and write geometry (Decode/Encode).
  //
  // The main drawback is that we need to go to some extra work to ensure that
  // "const" methods are still thread-safe.  Note that the goal is *not* to
  // make this class thread-safe in general, but simply to hide the fact that
  // we defer some of the indexing work until query time.
  //
  // The textbook approach to this problem would be to use a mutex and a
  // condition variable.  Unfortunately pthread mutexes are huge (40 bytes).
  // Instead we use spinlock (which is only 4 bytes) to guard a few small
  // fields representing the current update status, and only create additional
  // state while the update is actually occurring.
  mutable SpinLock lock_;

  enum IndexStatus {
    STALE,     // There are pending updates.
    UPDATING,  // Updates are currently being applied.
    FRESH,     // There are no pending updates.
  };
  // Reads and writes to this field are guarded by "lock_".
  std::atomic<IndexStatus> index_status_{FRESH};

  // UpdateState holds temporary data related to thread synchronization.  It
  // is only allocated while updates are being applied.
  struct UpdateState {
    // This mutex is used as a condition variable.  It is locked by the
    // updating thread for the entire duration of the update; other threads
    // lock it in order to wait until the update is finished.
    absl::Mutex wait_mutex;

    // The number of threads currently waiting on "wait_mutex_".  The
    // UpdateState can only be freed when this number reaches zero.
    //
    // Reads and writes to this field are guarded by "lock_".
    int num_waiting = 0;

    ~UpdateState() { ABSL_DCHECK_EQ(0, num_waiting); }
  };
  std::unique_ptr<UpdateState> update_state_;

  S2MemoryTracker::Client mem_tracker_;

#ifndef SWIG
  // Documented in the .cc file.
  void UnlockAndSignal() ABSL_UNLOCK_FUNCTION(lock_)
      ABSL_UNLOCK_FUNCTION(update_state_->wait_mutex);
#endif

  MutableS2ShapeIndex(const MutableS2ShapeIndex&) = delete;
  MutableS2ShapeIndex& operator=(const MutableS2ShapeIndex&) = delete;
};

// The following flag can be used to limit the amount of temporary memory used
// when building an S2ShapeIndex.  See the .cc file for details.
//
// DEFAULT: 100 MB
S2_DECLARE_int64(s2shape_index_tmp_memory_budget);


//////////////////   Implementation details follow   ////////////////////


// A BatchDescriptor represents a set of pending updates that will be applied
// at the same time.  The batch consists of all edges in (shape id, edge id)
// order from "begin" (inclusive) to "end" (exclusive).  Note that the last
// shape in a batch may have only some of its edges added.  The first batch
// also implicitly includes all shapes being removed.  "num_edges" is the
// total number of edges that will be added or removed in this batch.
struct MutableS2ShapeIndex::BatchDescriptor {
  // REQUIRES: If end.edge_id != 0, it must refer to a valid edge.
  ShapeEdgeId begin, end;
  int num_edges;
};

// The purpose of BatchGenerator is to divide large updates into batches such
// that all batches use approximately the same amount of high-water memory.
// This class is defined here so that it can be tested independently.
class MutableS2ShapeIndex::BatchGenerator {
 public:
  // Given the total number of edges that will be removed and added, prepares
  // to divide the edges into batches.  "shape_id_begin" identifies the first
  // shape whose edges will be added.
  BatchGenerator(int num_edges_removed, int num_edges_added,
                 int shape_id_begin);

  // Indicates that the given shape will be added to the index.  Shapes with
  // few edges will be grouped together into a single batch, while shapes with
  // many edges will be split over several batches if necessary.
  void AddShape(int shape_id, int num_edges);

  // Returns a vector describing each batch.  This method should be called
  // once all shapes have been added.
  std::vector<BatchDescriptor> Finish();

 private:
  // Returns a vector indicating the maximum number of edges in each batch.
  // (The actual batch sizes are adjusted later in order to avoid splitting
  // shapes between batches unnecessarily.)
  static std::vector<int> GetMaxBatchSizes(int num_edges_removed,
                                           int num_edges_added);

  // Returns the maximum number of edges in the current batch.
  int max_batch_size() const { return max_batch_sizes_[batch_index_]; }

  // Returns the maximum number of edges in the next batch.
  int next_max_batch_size() const { return max_batch_sizes_[batch_index_ + 1]; }

  // Adds the given number of edges to the current batch.
  void ExtendBatch(int num_edges) {
    batch_size_ += num_edges;
  }

  // Adds the given number of edges to the current batch, ending with the edge
  // just before "batch_end", and then starts a new batch.
  void FinishBatch(int num_edges, ShapeEdgeId batch_end);

  // A vector representing the ideal number of edges in each batch; the batch
  // sizes gradually decrease to ensure that each batch uses approximately the
  // same total amount of memory as the index grows.  The actual batch sizes
  // are then adjusted based on how many edges each shape has in order to
  // avoid splitting shapes between batches unnecessarily.
  std::vector<int> max_batch_sizes_;

  // The maximum size of the current batch is determined by how many edges
  // have been added to the index so far.  For example if GetBatchSizes()
  // returned {100, 70, 50, 30} and we have added 0 edges, the current batch
  // size is 100.  But if we have already added 90 edges then the current
  // batch size would be 70, and if have added 150 edges the batch size would
  // be 50.  We keep track of (1) the current index into batch_sizes and (2)
  // the number of edges remaining before we increment the batch index.
  int batch_index_ = 0;
  int batch_index_edges_left_ = 0;

  ShapeEdgeId batch_begin_;  // The start of the current batch.
  int shape_id_end_;         // One beyond the last shape to be added.
  int batch_size_ = 0;       // The number of edges in the current batch.
  std::vector<BatchDescriptor> batches_;  // The completed batches so far.
};

inline MutableS2ShapeIndex::Iterator::Iterator(
    const MutableS2ShapeIndex* index, InitialPosition pos) {
  Init(index, pos);
}

inline void MutableS2ShapeIndex::Iterator::Init(
    const MutableS2ShapeIndex* index, InitialPosition pos) {
  index->MaybeApplyUpdates();
  InitStale(index, pos);
}

inline void MutableS2ShapeIndex::Iterator::InitStale(
    const MutableS2ShapeIndex* index, InitialPosition pos) {
  index_ = index;
  end_ = index_->cell_map_.end();
  iter_ = end_;

  if (pos == BEGIN) {
    iter_ = index_->cell_map_.begin();
  }
}

inline void MutableS2ShapeIndex::Iterator::Begin() {
  // Make sure that the index has not been modified since Init() was called.
  ABSL_DCHECK(index_->is_fresh());
  iter_ = index_->cell_map_.begin();
}

inline void MutableS2ShapeIndex::Iterator::Finish() {
  iter_ = end_;
}

inline void MutableS2ShapeIndex::Iterator::Next() {
  ABSL_DCHECK(!done());
  ++iter_;
}

inline bool MutableS2ShapeIndex::Iterator::Prev() {
  if (iter_ == index_->cell_map_.begin()) {
    return false;
  }
  --iter_;
  return true;
}

inline void MutableS2ShapeIndex::Iterator::Seek(S2CellId target) {
  iter_ = index_->cell_map_.lower_bound(target);
}

inline std::unique_ptr<MutableS2ShapeIndex::IteratorBase>
MutableS2ShapeIndex::NewIterator(InitialPosition pos) const {
  return std::make_unique<Iterator>(this, pos);
}

inline void MutableS2ShapeIndex::ForceBuild() const {
  MaybeApplyUpdates();
}

inline bool MutableS2ShapeIndex::is_fresh() const {
  return index_status_.load(std::memory_order_relaxed) == FRESH;
}

// Given that the given shape is being updated, return true if it is being
// removed (as opposed to being added).
inline bool MutableS2ShapeIndex::is_shape_being_removed(int shape_id) const {
  // All shape ids being removed are less than all shape ids being added.
  return shape_id < pending_additions_begin_;
}

// Ensure that any pending updates have been applied.  This method must be
// called before accessing the cell_map_ field, even if the index_status_
// appears to be FRESH, because a memory barrier is required in order to
// ensure that all the index updates are visible if the updates were done in
// another thread.
inline void MutableS2ShapeIndex::MaybeApplyUpdates() const {
  // To avoid acquiring and releasing the spinlock on every query, we use
  // atomic operations when testing whether the status is FRESH and when
  // updating the status to be FRESH.  This guarantees that any thread that
  // sees a status of FRESH will also see the corresponding index updates.
  if (index_status_.load(std::memory_order_acquire) != FRESH) {
    const_cast<MutableS2ShapeIndex*>(this)->ApplyUpdatesThreadSafe();
  }
}

#endif  // S2_MUTABLE_S2SHAPE_INDEX_H_
