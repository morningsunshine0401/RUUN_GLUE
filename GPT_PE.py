import cv2
import torch
import numpy as np
from lightglue import LightGlue, SuperPoint
from KF_MK3 import MultExtendedKalmanFilter
from utils import rotation_matrix_to_quaternion, quaternion_to_rotation_matrix

class PoseEstimator:
    def __init__(self, opt, device, kf_mode=None):
        # 옵션 및 장치 설정
        self.device = device
        self.opt = opt
        
        # 앵커 이미지
        self.anchor_image = cv2.resize(cv2.imread(opt.anchor), tuple(opt.resize))
        
        # 앵커 2D/3D 키포인트 고정
        self.anchor_kpts2d = np.array([
            [511, 293], [591, 284], [587, 330], [413, 249], [602, 348],
            [715, 384], [598, 298], [656, 171], [805, 213], [703, 392],
            [523, 286], [519, 327], [387, 289], [727, 126], [425, 243],
            [636, 358], [745, 202], [595, 388], [436, 260], [539, 313],
            [795, 220], [351, 291], [665, 165], [611, 353], [650, 377],
            [516, 389], [727, 143], [496, 378], [575, 312], [617, 368],
            [430, 312], [480, 281], [834, 225], [469, 339], [705, 223],
            [637, 156], [816, 414], [357, 195], [752,  77], [642, 451]
        ], dtype=np.float32)
        self.anchor_kpts3d = np.array([
            [-0.014,  0.000,  0.042], [ 0.025, -0.014, -0.011],
            [-0.014,  0.000, -0.042], [-0.014,  0.000,  0.156],
            [-0.023,  0.000, -0.065], [ 0.000,  0.000, -0.156],
            [ 0.025,  0.000, -0.015], [ 0.217,  0.000,  0.070],
            [ 0.230,  0.000, -0.070], [-0.014,  0.000, -0.156],
            [ 0.000,  0.000,  0.042], [-0.057, -0.018, -0.010],
            [-0.074, -0.000,  0.128], [ 0.206, -0.070, -0.002],
            [-0.000, -0.000,  0.156], [-0.017, -0.000, -0.092],
            [ 0.217, -0.000, -0.027], [-0.052, -0.000, -0.097],
            [-0.019, -0.000,  0.128], [-0.035, -0.018, -0.010],
            [ 0.217, -0.000, -0.070], [-0.080, -0.000,  0.156],
            [ 0.230, -0.000,  0.070], [-0.023, -0.000, -0.075],
            [-0.029, -0.000, -0.127], [-0.090, -0.000, -0.042],
            [ 0.206, -0.055, -0.002], [-0.090, -0.000, -0.015],
            [ 0.000, -0.000, -0.015], [-0.037, -0.000, -0.097],
            [-0.074, -0.000,  0.074], [-0.019, -0.000,  0.074],
            [ 0.230, -0.000, -0.113], [-0.100, -0.030,  0.000],
            [ 0.170, -0.000, -0.015], [ 0.230, -0.000,  0.113],
            [-0.000, -0.025, -0.240], [-0.000, -0.025,  0.240],
            [ 0.243, -0.104,  0.000], [-0.080, -0.000, -0.156]
        ], dtype=np.float32)
        
        # SuperPoint + LightGlue 모델 초기화
        self.sp = SuperPoint(max_num_keypoints=512).eval().to(device)
        self.lg = LightGlue(features="superpoint").eval().to(device)
        
        # 앵커 특징 한 번만 추출
        with torch.no_grad():
            tensor = self._to_tensor(self.anchor_image)
            feats = self.sp.extract(tensor)
        self.anchor_feats = feats
        
        # Kalman 필터 상태
        self.mekf = None
        self.kf_initialized = False

    def process_frame(self, frame, frame_idx, bbox=None):
        # 크롭 & 전처리
        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]
        tensor = self._to_tensor(crop)

        # SuperPoint & 매칭
        with torch.no_grad():
            feats1 = self.sp.extract(tensor)
            matches = self.lg({ 'image0': self.anchor_feats, 'image1': feats1 })
        # frame keypoints 추출
        frame_kpts = feats1['keypoints'][0].cpu().numpy()
        mk0, mk1, pts3d = self._gather_correspondences(matches, frame_kpts)

        # PnP
        K, dist = self._get_camera_intrinsics()
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3d, mk1, K, dist, flags=cv2.SOLVEPNP_EPNP
        )
        if not ok or len(inliers) < 5:
            return None, crop

        # KF 초기화
        if not self.kf_initialized:
            self.mekf = MultExtendedKalmanFilter(dt=1/30.)
            x0 = np.zeros(self.mekf.n_states)
            x0[:3] = tvec.flatten()
            x0[6:10] = rotation_matrix_to_quaternion(cv2.Rodrigues(rvec)[0])
            self.mekf.x = x0
            self.kf_initialized = True
            return {'rvec': rvec, 'tvec': tvec}, crop

        # 예측 및 루즈리 커플 업데이트
        x_pred, _ = self.mekf.predict()
        q = rotation_matrix_to_quaternion(cv2.Rodrigues(rvec)[0])
        meas = np.concatenate([tvec.flatten(), q])
        x_upd, _ = self.mekf.update(meas)

        # 시각화
        vis = self._draw_axes(crop, rvec, tvec, x_upd)
        return {'kf_state': x_upd}, vis

    def _to_tensor(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.
        return torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(self.device)

        def _gather_correspondences(self, matches, frame_kpts):
        # matches0/matches1 배치 차원 제거 후 numpy 변환
        m0 = matches['matches0'][0].cpu().numpy()
        m1 = matches['matches1'][0].cpu().numpy()
        # 앵커 keypoints
        anchor_kpts = self.anchor_feats['keypoints'][0].cpu().numpy()
        # 유효 매칭 인덱스
        valid = m0 >= 0
        # 매칭된 2D pts
        mk0 = anchor_kpts[m0[valid]]
        mk1 = frame_kpts[m1[valid]]
        # 매칭된 3D pts
        pts3d = self.anchor_kpts3d[m0[valid]]
        return mk0, mk1, pts3d

    def _draw_axes(self, img, rvec, tvec, x_state):
        K, dist = self._get_camera_intrinsics()
        axes = np.float32([[0.1,0,0],[0,0.1,0],[0,0,0.1]]).reshape(-1,1,3)
        pts, _ = cv2.projectPoints(axes, rvec, tvec, K, dist)
        org = tuple(pts[0].ravel().astype(int))
        for p in pts[1:].reshape(-1,2):
            cv2.line(img, org, tuple(p.astype(int)), (0,0,255), 1)
        R = quaternion_to_rotation_matrix(x_state[6:10])
        rvec_kf, _ = cv2.Rodrigues(R)
        tvec_kf = x_state[:3].reshape(3,1)
        pts_kf, _ = cv2.projectPoints(axes, rvec_kf, tvec_kf, K, dist)
        org2 = tuple(pts_kf[0].ravel().astype(int))
        for p in pts_kf[1:].reshape(-1,2):
            cv2.line(img, org2, tuple(p.astype(int)), (0,255,0), 3)
        cv2.putText(img, f"KF pos: {x_state[:3]}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)
        return img

    def _get_camera_intrinsics(self):
        fx = 1460.10150
        fy = 1456.48915
        cx = 604.85462
        cy = 328.64800
        K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], np.float32)
        return K, None
