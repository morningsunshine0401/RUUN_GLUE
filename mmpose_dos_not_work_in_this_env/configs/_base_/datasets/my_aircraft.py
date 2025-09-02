# # File: configs/_base_/datasets/my_aircraft.py

# dataset_info = dict(
#     dataset_name='MatchaBox',
#     paper_info=dict(
#         author='RUUN',
#         title='My Aircraft Keypoint Dataset (8 KP)',
#         container='Roboflow Export',
#         year='2025',
#         homepage='https://public.roboflow.com/',
#     ),
#     keypoint_info={
#         0: dict(name='keypoint_0', id=0, color=[255, 0, 0],   type='upper', swap=''),
#         1: dict(name='keypoint_1', id=1, color=[0, 255, 0],   type='upper', swap=''),
#         2: dict(name='keypoint_2', id=2, color=[0, 0, 255],   type='upper', swap=''),
#         3: dict(name='keypoint_3', id=3, color=[255, 255, 0], type='lower', swap=''),
#         4: dict(name='keypoint_4', id=4, color=[255, 0, 255], type='lower', swap=''),
#         5: dict(name='keypoint_5', id=5, color=[0, 255, 255], type='lower', swap=''),
#         6: dict(name='keypoint_6', id=6, color=[128, 128, 0], type='lower', swap=''),
#         7: dict(name='keypoint_7', id=7, color=[0, 128, 128], type='lower', swap=''),
#     },
#     skeleton_info={
#         0: dict(link=('keypoint_0', 'keypoint_1'), id=0, color=[255, 0, 0]),
#         1: dict(link=('keypoint_1', 'keypoint_2'), id=1, color=[0, 255, 0]),
#         2: dict(link=('keypoint_2', 'keypoint_3'), id=2, color=[0, 0, 255]),
#         3: dict(link=('keypoint_3', 'keypoint_4'), id=3, color=[255, 255, 0]),
#         4: dict(link=('keypoint_4', 'keypoint_5'), id=4, color=[255, 0, 255]),
#         5: dict(link=('keypoint_5', 'keypoint_6'), id=5, color=[0, 255, 255]),
#         6: dict(link=('keypoint_6', 'keypoint_7'), id=6, color=[128, 128, 0]),
#         7: dict(link=('keypoint_7', 'keypoint_0'), id=7, color=[0, 128, 128]),
#     },
#     joint_weights=[1.0 for _ in range(8)],
#     sigmas=[0.1 for _ in range(8)],
#     num_keypoints=8,  # ✅ 꼭 명시해줘야 오류 방지 가능
# )



# File: configs/_base_/datasets/my_aircraft.py

dataset_info = dict(
    dataset_name='my_aircraft',
    paper_info=dict(
        author='RUUN',
        title='My Aircraft Keypoint Dataset',
        container='Roboflow Export',
        year='2025',
        homepage='https://public.roboflow.com/',
    ),
    # There are 6 keypoints in category_id=1 (“aircraft”).
    keypoint_info={
        0: dict(name='kp0', id=0, color=[255, 0, 0],   type='upper', swap=''), 
        1: dict(name='kp1', id=1, color=[0, 255, 0],   type='upper', swap=''), 
        2: dict(name='kp2', id=2, color=[0, 0, 255],   type='upper', swap=''), 
        3: dict(name='kp3', id=3, color=[255, 255, 0], type='lower', swap=''), 
        4: dict(name='kp4', id=4, color=[255, 0, 255], type='lower', swap=''), 
        5: dict(name='kp5', id=5, color=[0, 255, 255], type='lower', swap=''), 
    },
    # The original val.json used 1-based skeleton [[1,2],[2,3],[3,4],[4,5],[5,6],[6,1]].
    # Convert those to keypoint names and 0-based IDs here:
    skeleton_info={
        0: dict(link=('kp0', 'kp1'), id=0, color=[255, 0, 0]),
        1: dict(link=('kp1', 'kp2'), id=1, color=[0, 255, 0]),
        2: dict(link=('kp2', 'kp3'), id=2, color=[0, 0, 255]),
        3: dict(link=('kp3', 'kp4'), id=3, color=[255, 255, 0]),
        4: dict(link=('kp4', 'kp5'), id=4, color=[255, 0, 255]),
        5: dict(link=('kp5', 'kp0'), id=5, color=[0, 255, 255]),
    },
    # If you want uniform loss weighting, set all to 1.0.
    joint_weights=[1.0 for _ in range(6)],
    # Sigmas for OKS. Use a reasonable small sigma since keypoints are precise.
    sigmas=[0.1 for _ in range(6)],
)
