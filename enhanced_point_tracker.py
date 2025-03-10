import numpy as np
import logging

logger = logging.getLogger(__name__)

class EnhancedPointTracker:
    """
    Enhanced Point Tracker that combines SuperPoint features with temporal consistency.
    Maintains a sliding window of good features with 3D associations.
    """

    def __init__(self, max_length=5, nn_thresh=0.7, sliding_window_size=3):
        """
        Initialize the tracker with parameters.
        
        Args:
            max_length: Maximum track length to maintain
            nn_thresh: Nearest neighbor threshold for matching
            sliding_window_size: Number of frames to include in sliding window
        """
        if max_length < 2:
            raise ValueError('max_length must be greater than or equal to 2.')
        self.maxl = max_length
        self.nn_thresh = nn_thresh
        self.sliding_window_size = sliding_window_size
        
        # Store thresholds for quality metrics - these can be overridden by the caller
        self.inlier_ratio_threshold = 0.5
        self.reprojection_error_threshold = 5.0
        self.coverage_score_threshold = 0.4  # Lower threshold to make tracking more likely
        
        # Log initialization parameters
        logger.info(f"Initializing EnhancedPointTracker with parameters:")
        logger.info(f"  max_length: {max_length}")
        logger.info(f"  nn_thresh: {nn_thresh}")
        logger.info(f"  sliding_window_size: {sliding_window_size}")
        logger.info(f"  quality thresholds - IR: {self.inlier_ratio_threshold}, RE: {self.reprojection_error_threshold}, CS: {self.coverage_score_threshold}")
        
        # Initialize storage for points in sliding window
        self.all_pts = []
        self.all_descs = []
        for n in range(self.maxl):
            self.all_pts.append(np.zeros((3, 0)))  # [x, y, conf]
            self.all_descs.append(np.zeros((0, 0)))  # Descriptor dimensions will be set on first update
            
        # Track storage: [track_id, avg_score, point_id_0, ..., point_id_L-1]
        self.tracks = np.zeros((0, self.maxl + 2))
        self.track_count = 0
        self.max_score = 9999
        
        # Storage for 3D points associated with tracks
        self.track_3d_points = {}  # Map track_id -> 3D point
        
        # Tracking quality metrics
        self.quality_metrics = {
            'inlier_ratio': [],
            'reprojection_error': [],
            'coverage_score': []
        }
        self.quality_window_size = 10  # Store metrics for last 10 frames
        
        # Flag to check if initialized
        self.initialized = False
        self.desc_dim = None  # Set properly during first update

    def nn_match_two_way(self, desc1, desc2, nn_thresh):
        """
        Performs two-way nearest neighbor matching of two sets of descriptors.
        
        Args:
            desc1: Descriptor matrix for first set of points.
            desc2: Descriptor matrix for second set of points.
            nn_thresh: Descriptor distance below which is a good match.
            
        Returns:
            matches: 3xL numpy array of matches [idx1, idx2, score]
        """
        # Check if descriptors are valid
        if desc1 is None or desc2 is None or desc1.size == 0 or desc2.size == 0:
            logger.warning("Invalid descriptors for matching")
            return np.zeros((3, 0))

        # Convert descriptors to standard form if needed - we want DxN form (D=feature dim, N=num points)
        d1 = self._normalize_descriptor_matrix(desc1)
        d2 = self._normalize_descriptor_matrix(desc2)
        
        if d1 is None or d2 is None:
            logger.warning("Could not normalize descriptor matrices for matching")
            return np.zeros((3, 0))
        
        # Compute descriptor distance matrix
        try:
            # Normalize descriptor vectors for better matching
            d1_norm = d1 / (np.linalg.norm(d1, axis=0, keepdims=True) + 1e-8)
            d2_norm = d2 / (np.linalg.norm(d2, axis=0, keepdims=True) + 1e-8)
            
            # Compute similarity (cosine) - higher is better
            similarity = np.dot(d1_norm.T, d2_norm)
            
            # Convert to distance (lower is better)
            dmat = np.sqrt(2 - 2 * np.clip(similarity, -1, 1))
            
            # Get NN indices and scores
            idx = np.argmin(dmat, axis=1)
            scores = dmat[np.arange(dmat.shape[0]), idx]
            
            # Threshold the matches
            keep = scores < nn_thresh
            
            # Check if nearest neighbor goes both ways
            idx2 = np.argmin(dmat, axis=0)
            keep_bi = np.arange(len(idx)) == idx2[idx]
            keep = np.logical_and(keep, keep_bi)
            
            # Format matches as [idx1, idx2, score]
            m_idx1 = np.arange(d1.shape[1])[keep]
            m_idx2 = idx[keep]
            scores = scores[keep]
            
            matches = np.zeros((3, int(keep.sum())))
            matches[0, :] = m_idx1
            matches[1, :] = m_idx2
            matches[2, :] = scores
            
            return matches
            
        except Exception as e:
            logger.error(f"Error in nn_match_two_way: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return np.zeros((3, 0))

    def _normalize_descriptor_matrix(self, desc):
        """
        Normalize descriptor matrix to standard DxN form
        where D is descriptor dimension and N is number of points.
        
        Args:
            desc: Descriptor matrix in either DxN or NxD form
            
        Returns:
            desc_normalized: Descriptor matrix in DxN form
        """
        if desc is None or desc.size == 0:
            return None
            
        # If it's a 1D array, reshape to 2D
        if len(desc.shape) == 1:
            # Assume it's a single descriptor
            return desc.reshape(-1, 1)
            
        # Determine if descriptor is in DxN or NxD form
        # SuperPoint descriptors are 256-dimensional
        # We'll use that as a heuristic - if either dimension is 256 or close to it, that's D
        if desc.shape[0] <= desc.shape[1] and desc.shape[0] > 200:
            # Already in DxN form (fewer rows than columns, and rows ~= 256)
            return desc
        elif desc.shape[1] <= desc.shape[0] and desc.shape[1] > 200:
            # In NxD form (fewer columns than rows, and columns ~= 256)
            return desc.T
        else:
            # If neither dimension is near 256, use the larger dimension as D
            if desc.shape[0] >= desc.shape[1]:
                # More rows than columns, assume NxD form
                return desc.T
            else:
                # More columns than rows, assume DxN form
                return desc

    def get_offsets(self):
        """
        Compute offsets for indexing into the list of points.
        
        Returns:
            offsets: N length array with integer offset locations.
        """
        offsets = [0]
        for i in range(len(self.all_pts) - 1):
            offsets.append(self.all_pts[i].shape[1])
        offsets = np.array(offsets)
        offsets = np.cumsum(offsets)
        return offsets

    def update(self, pts, desc, pts3d=None):
        """
        Add a new set of points and descriptors to the tracker.
        
        Args:
            pts: 3xN numpy array of point observations [x, y, confidence].
            desc: Descriptor matrix for the points.
            pts3d: list or array of corresponding 3D points.
            
        Returns:
            tracked_pts3d: Dictionary mapping point indices to 3D points.
        """
        if pts is None or desc is None or pts.size == 0 or desc.size == 0:
            logger.warning('EnhancedPointTracker: Warning, no points were added to tracker.')
            return {}
            
        # Normalize the descriptor matrix to DxN form
        desc_normalized = self._normalize_descriptor_matrix(desc)
        if desc_normalized is None:
            logger.warning("Could not normalize descriptor matrix")
            return {}
        
        # Initialize descriptor dimension on first update
        if not self.initialized:
            self.desc_dim = desc_normalized.shape[0]
            # Reinitialize descriptor storage with correct dimensions
            self.all_descs = []
            for n in range(self.maxl):
                self.all_descs.append(np.zeros((self.desc_dim, 0)))
            self.initialized = True
            logger.info(f"Tracker initialized with descriptor dimension {self.desc_dim}")
        
        # Check point and descriptor counts match
        if pts.shape[1] != desc_normalized.shape[1]:
            logger.warning(f"Point count ({pts.shape[1]}) doesn't match descriptor count ({desc_normalized.shape[1]})")
            return {}
            
        # Remove oldest points and descriptors
        remove_size = self.all_pts[0].shape[1]
        self.all_pts.pop(0)
        self.all_pts.append(pts)
        
        self.all_descs.pop(0)
        self.all_descs.append(desc_normalized)
        
        # Remove oldest point in track
        self.tracks = np.delete(self.tracks, 2, axis=1)
        
        # Update track offsets
        for i in range(2, self.tracks.shape[1]):
            self.tracks[:, i] -= remove_size
        self.tracks[:, 2:][self.tracks[:, 2:] < -1] = -1
        
        offsets = self.get_offsets()
        
        # Add a new -1 column
        self.tracks = np.hstack((self.tracks, -1 * np.ones((self.tracks.shape[0], 1))))
        
        # Try to match to existing tracks
        matched = np.zeros((pts.shape[1])).astype(bool)
        tracked_pts3d = {}
        
        # Match with the most recent frame if we have previous frames
        if len(self.all_descs) > 1 and self.all_descs[-2].shape[1] > 0:
            try:
                # Match between last frame and current frame
                matches = self.nn_match_two_way(self.all_descs[-2], desc_normalized, self.nn_thresh)
                
                for match in matches.T:
                    # Add a new point to its matched track
                    id1 = int(match[0]) + offsets[-2]
                    id2 = int(match[1]) + offsets[-1]
                    found = np.argwhere(self.tracks[:, -2] == id1)
                    
                    if found.shape[0] > 0:
                        matched[int(match[1])] = True
                        row = int(found)
                        self.tracks[row, -1] = id2
                        
                        # Update track score with running average
                        if self.tracks[row, 1] == self.max_score:
                            self.tracks[row, 1] = match[2]
                        else:
                            track_len = (self.tracks[row, 2:] != -1).sum() - 1.
                            frac = 1. / float(max(track_len, 1))  # Avoid division by zero
                            self.tracks[row, 1] = (1.-frac)*self.tracks[row, 1] + frac*match[2]
                        
                        # Propagate 3D point association
                        track_id = int(self.tracks[row, 0])
                        if track_id in self.track_3d_points:
                            tracked_pts3d[int(match[1])] = self.track_3d_points[track_id]
            except Exception as e:
                logger.error(f"Error matching descriptors: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Add new tracks for unmatched points
        new_ids = np.arange(pts.shape[1]) + offsets[-1]
        new_ids = new_ids[~matched]
        new_tracks = -1 * np.ones((new_ids.shape[0], self.maxl + 2))
        new_tracks[:, -1] = new_ids
        new_tracks[:, 0] = self.track_count + np.arange(new_ids.shape[0])
        new_tracks[:, 1] = self.max_score * np.ones(new_ids.shape[0])
        self.tracks = np.vstack((self.tracks, new_tracks))
        
        # Update track count
        new_track_count = self.track_count + new_ids.shape[0]
        
        # If 3D points are provided, associate them with new and matched points
        if pts3d is not None:
            if isinstance(pts3d, list):
                pts3d = np.array(pts3d)
                
            if len(pts3d) != pts.shape[1]:
                logger.warning(f"3D points count ({len(pts3d)}) doesn't match 2D points ({pts.shape[1]})")
            else:
                # Associate 3D points with all points - matched points already handled above
                for i, pt_idx in enumerate(np.where(~matched)[0]):
                    track_id = self.track_count + i
                    self.track_3d_points[track_id] = pts3d[pt_idx]
        
        self.track_count = new_track_count
        
        # Remove empty tracks
        keep_rows = np.any(self.tracks[:, 2:] >= 0, axis=1)
        self.tracks = self.tracks[keep_rows, :]
        
        return tracked_pts3d

    def update_quality_metrics(self, inlier_ratio, reprojection_error, coverage_score):
        """
        Update tracking quality metrics.
        
        Args:
            inlier_ratio: Ratio of inliers to total matches
            reprojection_error: Mean reprojection error
            coverage_score: Coverage score [0-1]
        """
        self.quality_metrics['inlier_ratio'].append(inlier_ratio)
        self.quality_metrics['reprojection_error'].append(reprojection_error)
        self.quality_metrics['coverage_score'].append(coverage_score)
        
        # Trim to window size
        for key in self.quality_metrics:
            if len(self.quality_metrics[key]) > self.quality_window_size:
                self.quality_metrics[key] = self.quality_metrics[key][-self.quality_window_size:]

    def get_quality_trend(self):
        """
        Analyze tracking quality trend to determine if reinitialization is needed.
        
        Returns:
            quality_ok: Boolean indicating if tracking quality is acceptable
            metrics: Dictionary with current quality metrics
        """
        # Compute average metrics for the window
        metrics = {}
        for key in self.quality_metrics:
            if len(self.quality_metrics[key]) > 0:
                metrics[key] = np.mean(self.quality_metrics[key])
            else:
                metrics[key] = 0.0
        
        # Check if quality is acceptable with detailed logging
        inlier_ratio = metrics.get('inlier_ratio', 0)
        reprojection_error = metrics.get('reprojection_error', 100)
        coverage_score = metrics.get('coverage_score', 0)
        
        logger.debug(f"Quality metrics - IR: {inlier_ratio:.3f}/{self.inlier_ratio_threshold:.3f}, " +
                    f"RE: {reprojection_error:.3f}/{self.reprojection_error_threshold:.3f}, " +
                    f"CS: {coverage_score:.3f}/{self.coverage_score_threshold:.3f}")
        
        quality_ok = (
            inlier_ratio >= self.inlier_ratio_threshold and
            reprojection_error <= self.reprojection_error_threshold and
            coverage_score >= self.coverage_score_threshold
        )
        
        return quality_ok, metrics

    def get_tracks(self, min_length):
        """
        Retrieve point tracks of a given minimum length.
        
        Args:
            min_length: Minimum track length
            
        Returns:
            tracks: M x (2+L) sized matrix of tracks
        """
        if min_length < 1:
            raise ValueError('\'min_length\' too small.')
            
        valid = np.ones((self.tracks.shape[0])).astype(bool)
        good_len = np.sum(self.tracks[:, 2:] != -1, axis=1) >= min_length
        not_headless = (self.tracks[:, -1] != -1)
        keepers = np.logical_and.reduce((valid, good_len, not_headless))
        
        return self.tracks[keepers, :].copy()

    def get_active_tracks_with_3d(self):
        """
        Get active tracks with associated 3D points.
        
        Returns:
            tracks_2d: Nx2 array of 2D points
            tracks_3d: Nx3 array of 3D points
        """
        active_tracks = self.get_tracks(min_length=2)
        
        tracks_2d = []
        tracks_3d = []
        
        if active_tracks.shape[0] == 0:
            return np.array(tracks_2d), np.array(tracks_3d)
            
        # Get the most recent point of each track
        offsets = self.get_offsets()
        
        for track in active_tracks:
            track_id = int(track[0])
            last_pt_id = int(track[-1])
            
            if track_id in self.track_3d_points and last_pt_id >= 0:
                # Find which frame this point belongs to
                for i in range(len(offsets)-1, -1, -1):
                    if last_pt_id >= offsets[i]:
                        frame_idx = i
                        pt_idx = last_pt_id - offsets[i]
                        break
                
                if frame_idx < len(self.all_pts) and pt_idx < self.all_pts[frame_idx].shape[1]:
                    pt = self.all_pts[frame_idx][:, pt_idx]
                    tracks_2d.append(pt[:2])  # Use only x, y
                    tracks_3d.append(self.track_3d_points[track_id])
        
        return np.array(tracks_2d), np.array(tracks_3d)

# import numpy as np
# import logging

# logger = logging.getLogger(__name__)

# class EnhancedPointTracker:
#     """
#     Enhanced Point Tracker that combines SuperPoint features with temporal consistency.
#     Maintains a sliding window of good features with 3D associations.
#     """

#     def __init__(self, max_length=5, nn_thresh=0.7, sliding_window_size=3):
#         """
#         Initialize the tracker with parameters.
        
#         Args:
#             max_length: Maximum track length to maintain
#             nn_thresh: Nearest neighbor threshold for matching
#             sliding_window_size: Number of frames to include in sliding window
#         """
#         if max_length < 2:
#             raise ValueError('max_length must be greater than or equal to 2.')
#         self.maxl = max_length
#         self.nn_thresh = nn_thresh
#         self.sliding_window_size = sliding_window_size

#         # Store thresholds for quality metrics - these can be overridden by the caller
#         self.inlier_ratio_threshold = 0.5
#         self.reprojection_error_threshold = 5.0
#         self.coverage_score_threshold = 0.4  # Lower threshold to make tracking more likely
        
#         # Log initialization parameters
#         logger.info(f"Initializing EnhancedPointTracker with parameters:")
#         logger.info(f"  max_length: {max_length}")
#         logger.info(f"  nn_thresh: {nn_thresh}")
#         logger.info(f"  sliding_window_size: {sliding_window_size}")
#         logger.info(f"  quality thresholds - IR: {self.inlier_ratio_threshold}, RE: {self.reprojection_error_threshold}, CS: {self.coverage_score_threshold}")
        
            
#         # Initialize storage for points in sliding window
#         self.all_pts = []
#         self.all_descs = []
#         for n in range(self.maxl):
#             self.all_pts.append(np.zeros((3, 0)))  # [x, y, conf]
#             self.all_descs.append(np.zeros((0, 0)))  # Descriptor dimensions will be set on first update
            
#         # Track storage: [track_id, avg_score, point_id_0, ..., point_id_L-1]
#         self.tracks = np.zeros((0, self.maxl + 2))
#         self.track_count = 0
#         self.max_score = 9999
        
#         # Storage for 3D points associated with tracks
#         self.track_3d_points = {}  # Map track_id -> 3D point
        
#         # Tracking quality metrics
#         self.quality_metrics = {
#             'inlier_ratio': [],
#             'reprojection_error': [],
#             'coverage_score': []
#         }
#         self.quality_window_size = 10  # Store metrics for last 10 frames
        
#         # Flag to check if initialized
#         self.initialized = False
#         self.desc_dim = None  # Will be set on first update

#     def nn_match_two_way(self, desc1, desc2, nn_thresh):
#         """
#         Performs two-way nearest neighbor matching of two sets of descriptors.
        
#         Args:
#             desc1: DxN numpy matrix of D-dimensional descriptors for N points.
#             desc2: DxM numpy matrix of D-dimensional descriptors for M points.
#             nn_thresh: Descriptor distance below which is a good match.
            
#         Returns:
#             matches: 3xL numpy array of matches [idx1, idx2, score]
#         """
#         # Check if descriptors are valid
#         if desc1.shape[1] == 0 or desc2.shape[1] == 0:
#             return np.zeros((3, 0))
        
#         # Ensure descriptor dimensions match
#         if desc1.shape[0] != desc2.shape[0]:
#             logger.warning(f"Descriptor dimensions don't match: {desc1.shape} vs {desc2.shape}")
#             # Try to fix by transposing if needed
#             if desc1.shape[0] == desc2.shape[1] and desc1.shape[1] == desc2.shape[0]:
#                 logger.warning("Transposing descriptor matrix to match dimensions")
#                 desc2 = desc2.T
#             else:
#                 # Cannot match, return empty
#                 return np.zeros((3, 0))
            
#         # Compute descriptor distance matrix
#         dmat = np.dot(desc1.T, desc2)
#         dmat = np.sqrt(2 - 2 * np.clip(dmat, -1, 1))
        
#         # Get NN indices and scores
#         idx = np.argmin(dmat, axis=1)
#         scores = dmat[np.arange(dmat.shape[0]), idx]
        
#         # Threshold the matches
#         keep = scores < nn_thresh
        
#         # Check if nearest neighbor goes both ways
#         idx2 = np.argmin(dmat, axis=0)
#         keep_bi = np.arange(len(idx)) == idx2[idx]
#         keep = np.logical_and(keep, keep_bi)
        
#         # Format matches as [idx1, idx2, score]
#         m_idx1 = np.arange(desc1.shape[1])[keep]
#         m_idx2 = idx[keep]
#         scores = scores[keep]
        
#         matches = np.zeros((3, int(keep.sum())))
#         matches[0, :] = m_idx1
#         matches[1, :] = m_idx2
#         matches[2, :] = scores
        
#         return matches

#     def get_offsets(self):
#         """
#         Compute offsets for indexing into the list of points.
        
#         Returns:
#             offsets: N length array with integer offset locations.
#         """
#         offsets = [0]
#         for i in range(len(self.all_pts) - 1):
#             offsets.append(self.all_pts[i].shape[1])
#         offsets = np.array(offsets)
#         offsets = np.cumsum(offsets)
#         return offsets

#     def update(self, pts, desc, pts3d=None):
#         """
#         Add a new set of points and descriptors to the tracker.
        
#         Args:
#             pts: 3xN numpy array of point observations [x, y, confidence].
#             desc: DxN numpy array of corresponding D dimensional descriptors.
#             pts3d: list or array of corresponding 3D points.
            
#         Returns:
#             tracked_pts3d: Dictionary mapping point indices to 3D points.
#         """
#         if pts is None or desc is None:
#             logger.warning('EnhancedPointTracker: Warning, no points were added to tracker.')
#             return {}
        
#         # Check dimensions    
#         if pts.shape[1] != desc.shape[1]:
#             logger.warning(f"Point and descriptor counts don't match: {pts.shape[1]} vs {desc.shape[1]}")
#             # Try to fix by transposing descriptor if shapes suggest it
#             if pts.shape[1] == desc.shape[0] and desc.shape[1] > pts.shape[1]:
#                 logger.warning("Transposing descriptor matrix to match point count")
#                 desc = desc.T
            
#             # Check again after potential fix
#             if pts.shape[1] != desc.shape[1]:
#                 logger.error("Cannot reconcile point and descriptor counts")
#                 return {}
        
#         # Initialize descriptor dimension on first update
#         if not self.initialized:
#             self.desc_dim = desc.shape[0]
#             # Reinitialize descriptor storage with correct dimensions
#             self.all_descs = []
#             for n in range(self.maxl):
#                 self.all_descs.append(np.zeros((self.desc_dim, 0)))
#             self.initialized = True
#             logger.info(f"Tracker initialized with descriptor dimension {self.desc_dim}")
        
#         # Ensure descriptor dimension matches what we expect
#         if desc.shape[0] != self.desc_dim:
#             logger.warning(f"Descriptor dimension changed: expected {self.desc_dim}, got {desc.shape[0]}")
#             # Try to fix by transposing
#             if desc.shape[1] == self.desc_dim:
#                 logger.warning("Transposing descriptor matrix to match expected dimension")
#                 desc = desc.T
            
#             # Check again after potential fix
#             if desc.shape[0] != self.desc_dim:
#                 logger.error("Cannot reconcile descriptor dimensions")
#                 return {}
        
#         # Remove oldest points and descriptors
#         remove_size = self.all_pts[0].shape[1]
#         self.all_pts.pop(0)
#         self.all_pts.append(pts)
        
#         self.all_descs.pop(0)
#         self.all_descs.append(desc)
        
#         # Remove oldest point in track
#         self.tracks = np.delete(self.tracks, 2, axis=1)
        
#         # Update track offsets
#         for i in range(2, self.tracks.shape[1]):
#             self.tracks[:, i] -= remove_size
#         self.tracks[:, 2:][self.tracks[:, 2:] < -1] = -1
        
#         offsets = self.get_offsets()
        
#         # Add a new -1 column
#         self.tracks = np.hstack((self.tracks, -1 * np.ones((self.tracks.shape[0], 1))))
        
#         # Try to match to existing tracks
#         matched = np.zeros((pts.shape[1])).astype(bool)
#         tracked_pts3d = {}
        
#         # Match with the most recent frame if we have previous frames
#         if len(self.all_descs) > 1 and self.all_descs[-2].shape[1] > 0:
#             try:
#                 matches = self.nn_match_two_way(self.all_descs[-2], desc, self.nn_thresh)
                
#                 for match in matches.T:
#                     # Add a new point to its matched track
#                     id1 = int(match[0]) + offsets[-2]
#                     id2 = int(match[1]) + offsets[-1]
#                     found = np.argwhere(self.tracks[:, -2] == id1)
                    
#                     if found.shape[0] > 0:
#                         matched[int(match[1])] = True
#                         row = int(found)
#                         self.tracks[row, -1] = id2
                        
#                         # Update track score with running average
#                         if self.tracks[row, 1] == self.max_score:
#                             self.tracks[row, 1] = match[2]
#                         else:
#                             track_len = (self.tracks[row, 2:] != -1).sum() - 1.
#                             frac = 1. / float(max(track_len, 1))  # Avoid division by zero
#                             self.tracks[row, 1] = (1.-frac)*self.tracks[row, 1] + frac*match[2]
                        
#                         # Propagate 3D point association
#                         track_id = int(self.tracks[row, 0])
#                         if track_id in self.track_3d_points:
#                             tracked_pts3d[int(match[1])] = self.track_3d_points[track_id]
#             except Exception as e:
#                 logger.error(f"Error matching descriptors: {e}")
        
#         # Add new tracks for unmatched points
#         new_ids = np.arange(pts.shape[1]) + offsets[-1]
#         new_ids = new_ids[~matched]
#         new_tracks = -1 * np.ones((new_ids.shape[0], self.maxl + 2))
#         new_tracks[:, -1] = new_ids
#         new_tracks[:, 0] = self.track_count + np.arange(new_ids.shape[0])
#         new_tracks[:, 1] = self.max_score * np.ones(new_ids.shape[0])
#         self.tracks = np.vstack((self.tracks, new_tracks))
        
#         # Update track count
#         new_track_count = self.track_count + new_ids.shape[0]
        
#         # If 3D points are provided, associate them with new tracks
#         if pts3d is not None:
#             if len(pts3d) != pts.shape[1]:
#                 logger.warning(f"3D points count doesn't match 2D points: {len(pts3d)} vs {pts.shape[1]}")
            
#             # Associate 3D points with unmatched points
#             for i, pt_idx in enumerate(np.where(~matched)[0]):
#                 if pt_idx < len(pts3d):
#                     track_id = self.track_count + i
#                     self.track_3d_points[track_id] = pts3d[pt_idx]
        
#         self.track_count = new_track_count
        
#         # Remove empty tracks
#         keep_rows = np.any(self.tracks[:, 2:] >= 0, axis=1)
#         self.tracks = self.tracks[keep_rows, :]
        
#         return tracked_pts3d

#     def update_quality_metrics(self, inlier_ratio, reprojection_error, coverage_score):
#         """
#         Update tracking quality metrics.
        
#         Args:
#             inlier_ratio: Ratio of inliers to total matches
#             reprojection_error: Mean reprojection error
#             coverage_score: Coverage score [0-1]
#         """
#         self.quality_metrics['inlier_ratio'].append(inlier_ratio)
#         self.quality_metrics['reprojection_error'].append(reprojection_error)
#         self.quality_metrics['coverage_score'].append(coverage_score)
        
#         # Trim to window size
#         for key in self.quality_metrics:
#             if len(self.quality_metrics[key]) > self.quality_window_size:
#                 self.quality_metrics[key] = self.quality_metrics[key][-self.quality_window_size:]

#     # def get_quality_trend(self):
#     #     """
#     #     Analyze tracking quality trend to determine if reinitialization is needed.
        
#     #     Returns:
#     #         quality_ok: Boolean indicating if tracking quality is acceptable
#     #         metrics: Dictionary with current quality metrics
#     #     """
#     #     # Compute average metrics for the window
#     #     metrics = {}
#     #     for key in self.quality_metrics:
#     #         if len(self.quality_metrics[key]) > 0:
#     #             metrics[key] = np.mean(self.quality_metrics[key])
#     #         else:
#     #             metrics[key] = 0.0
        
#     #     # Define quality thresholds
#     #     inlier_ratio_threshold = 0.5
#     #     reprojection_error_threshold = 5.0
#     #     coverage_score_threshold = 0.6
        
#     #     # Check if quality is acceptable
#     #     quality_ok = (
#     #         metrics.get('inlier_ratio', 0) > inlier_ratio_threshold and
#     #         metrics.get('reprojection_error', 100) < reprojection_error_threshold and
#     #         metrics.get('coverage_score', 0) > coverage_score_threshold
#     #     )
        
#     #     return quality_ok, metrics
#     def get_quality_trend(self):
#         """
#         Analyze tracking quality trend to determine if reinitialization is needed.
        
#         Returns:
#             quality_ok: Boolean indicating if tracking quality is acceptable
#             metrics: Dictionary with current quality metrics
#         """
#         # Compute average metrics for the window
#         metrics = {}
#         for key in self.quality_metrics:
#             if len(self.quality_metrics[key]) > 0:
#                 metrics[key] = np.mean(self.quality_metrics[key])
#             else:
#                 metrics[key] = 0.0
        
#         # Define quality thresholds - use instance variables if available
#         # (This helps with lowering thresholds via command line)
#         if hasattr(self, 'inlier_ratio_threshold'):
#             inlier_ratio_threshold = self.inlier_ratio_threshold
#         else:
#             inlier_ratio_threshold = 0.5
            
#         if hasattr(self, 'reprojection_error_threshold'):
#             reprojection_error_threshold = self.reprojection_error_threshold
#         else:
#             reprojection_error_threshold = 5.0
            
#         if hasattr(self, 'coverage_score_threshold'):
#             coverage_score_threshold = self.coverage_score_threshold
#         else: 
#             coverage_score_threshold = 0.4  # Lower default threshold
        
#         # Check if quality is acceptable with detailed logging
#         inlier_ratio = metrics.get('inlier_ratio', 0)
#         reprojection_error = metrics.get('reprojection_error', 100)
#         coverage_score = metrics.get('coverage_score', 0)
        
#         logger.debug(f"Quality metrics - IR: {inlier_ratio:.3f}/{inlier_ratio_threshold:.3f}, " +
#                     f"RE: {reprojection_error:.3f}/{reprojection_error_threshold:.3f}, " +
#                     f"CS: {coverage_score:.3f}/{coverage_score_threshold:.3f}")
        
#         quality_ok = (
#             inlier_ratio >= inlier_ratio_threshold and
#             reprojection_error <= reprojection_error_threshold and
#             coverage_score >= coverage_score_threshold
#         )
        
#         return quality_ok, metrics


#     def get_tracks(self, min_length):
#         """
#         Retrieve point tracks of a given minimum length.
        
#         Args:
#             min_length: Minimum track length
            
#         Returns:
#             tracks: M x (2+L) sized matrix of tracks
#         """
#         if min_length < 1:
#             raise ValueError('\'min_length\' too small.')
            
#         valid = np.ones((self.tracks.shape[0])).astype(bool)
#         good_len = np.sum(self.tracks[:, 2:] != -1, axis=1) >= min_length
#         not_headless = (self.tracks[:, -1] != -1)
#         keepers = np.logical_and.reduce((valid, good_len, not_headless))
        
#         return self.tracks[keepers, :].copy()

#     def get_active_tracks_with_3d(self):
#         """
#         Get active tracks with associated 3D points.
        
#         Returns:
#             tracks_2d: Nx2 array of 2D points
#             tracks_3d: Nx3 array of 3D points
#         """
#         active_tracks = self.get_tracks(min_length=2)
        
#         tracks_2d = []
#         tracks_3d = []
        
#         if active_tracks.shape[0] == 0:
#             return np.array(tracks_2d), np.array(tracks_3d)
            
#         # Get the most recent point of each track
#         offsets = self.get_offsets()
        
#         for track in active_tracks:
#             track_id = int(track[0])
#             last_pt_id = int(track[-1])
            
#             if track_id in self.track_3d_points and last_pt_id >= 0:
#                 # Find which frame this point belongs to
#                 for i in range(len(offsets)-1, -1, -1):
#                     if last_pt_id >= offsets[i]:
#                         frame_idx = i
#                         pt_idx = last_pt_id - offsets[i]
#                         break
                
#                 if frame_idx < len(self.all_pts) and pt_idx < self.all_pts[frame_idx].shape[1]:
#                     pt = self.all_pts[frame_idx][:, pt_idx]
#                     tracks_2d.append(pt[:2])  # Use only x, y
#                     tracks_3d.append(self.track_3d_points[track_id])
        
#         return np.array(tracks_2d), np.array(tracks_3d)