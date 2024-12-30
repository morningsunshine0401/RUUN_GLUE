-moving pnp cmd
	 python3 RuunPose-3D.py --anchor assets/Ruun_images/boxcraft/frame_00001.png --resize 640 480 --superglue outdoor 

python3 RuunPose-3D-video-research.py --anchor assets/Ruun_images/boxcraft/frame_00001.png --resize 640 480 --superglue outdoor --input Ruun_code/20241002_output_translation.avi --nms_radius 3 --show_keypoints --match_threshold 0.3 





python3 RuunPose-3D-video-research-blender.py --anchor assets/Ruun_images/boxcraft/blender/00010002.png --resize 640 480 --superglue outdoor --ground_truth assets/Ruun_images/boxcraft/blender/object_poses.json --input assets/Ruun_images/boxcraft/blender/

python3 RuunPose-3D-video-research-blender.py --input assets/Ruun_images/cube/ --anchor assets/Ruun_images/cube/0002.png --ground_truth assets/Ruun_images/cube/weird_cube_object_poses.json --resize 640 480 --superglue outdoor


python3 RuunPose-3D-video-research-blender.py --input assets/Ruun_images/cube/better_frame/ --anchor assets/Ruun_images/cube/better_frame/0002.png --ground_truth assets/Ruun_images/cube/better_frame/weird_cube_object_poses_1280.json --resize 1280 960 --superglue outdoor


python3 RuunPose-3D-viewpoint-blender.py --input assets/Ruun_images/viewpoint/test/ --ground_truth assets/Ruun_images/viewpoint/test/viewpoint_GT.json --resize 1280 960 --superglue outdoor --show_keypoints --viewpoint_model_path viewpoint_model_no_aug.pth 

python3 RuunPose-3D-viewpoint-blender.py --input assets/Ruun_images/viewpoint/test/ideal/ --ground_truth assets/Ruun_images/viewpoint/test/ideal/viewpoint_GT.json --resize 1280 960 --superglue outdoor --show_keypoints --viewpoint_model_path viewpoint_model_no_aug.pth 

python3 RuunPose-3D-viewpoint-blender.py --input assets/Ruun_images/viewpoint/test/ideal/ --ground_truth assets/Ruun_images/viewpoint/test/ideal/viewpoint_GT.json --resize 1280 960 --superglue outdoor --show_keypoints --viewpoint_model_path viewpoint_model_no_aug.pth --output_dir dump_match_pairs/match_output/viewpoint/result/ --show_keypoints

python3 RuunPose-3D-viewpoint-blender.py --input assets/Ruun_images/viewpoint/test/rotated/ --ground_truth assets/Ruun_images/viewpoint/test/rotated/viewpoint_GT_rotate.json --resize 1280 960 --superglue outdoor --show_keypoints --viewpoint_model_path viewpoint_model_more_data.pth --output_dir dump_match_pairs/match_output/viewpoint/result/rotate/ --show_keypoints


python3 RuunPose-3D-viewpoint-blender.py --input assets/Ruun_images/viewpoint/test/ --ground_truth assets/Ruun_images/viewpoint/test/viewpoint_GT.json --resize 1280 960 --superglue outdoor --show_keypoints --viewpoint_model_path viewpoint_model_more_data.pth --output_dir dump_match_pairs/match_output/viewpoint/result/ --show_keypoints --match_threshold 0.1 



python3 demo_superglue.py --input assets/Sky_test/N/ --output RuunPoseResult/20241106_result/N/ --superglue outdoor --resize 1280 960 --show_keypoints --keypoint_threshold 0.05


./demo_superglue.py --input assets/Ruun_images/viewpoint/anchor/anchor_data/1 --output_dir assets/Ruun_images/viewpoint/anchor/anchor_data/1_result/ --resize 1280 720 --no_display --superglue outdoor --show_keypoints --keypoint_threshold 0.02 -h


202241118 python3 match_pairs.py --output_dir assets/Ruun_images/viewpoint/anchor/anchor_data/1_result/ --resize 1280 720 --superglue outdoor --show_keypoints --keypoint_threshold 0.04 --input_pair assets/scannet_sample_pairs_with_gt.txt --input_dir assets/Ruun_images/viewpoint/anchor/anchor_data/1/


1.png 30.png 0 0 1777 0 640 0 1111 400 0 0 1 1777 0 640 0 1111 400 0 0 1 0.9217 0.1926 -0.3367 -1.8852 -0.1671 0.9805 0.1036 -5.9441 0.3500 -0.0393 0.9359 2.1322 0 0 0 1



python3 20241119_RuunPose-realtime.py --anchor assets/Ruun_images/viewpoint/anchor/realAnchor.png --resize 1280 720 --superglue outdoor --input Ruun_code/back.avi --nms_radius 3 --show_keypoints --match_threshold 0.3

python3 20241119_RuunPose-realtime.py --anchor assets/Ruun_images/viewpoint/anchor/realAnchor.png --resize 1280 720 --superglue outdoor --input Ruun_code/aircraft_test.avi --nms_radius 3 --show_keypoints --match_threshold 0.3 --keypoint_threshold 0.02 --output_dir dump_match_pairs/20241119/ --no_display 

python3 20241119_RuunPose-realtime.py --anchor assets/Ruun_images/viewpoint/anchor/realAnchor.png --resize 1280 720 --superglue outdoor --nms_radius 3 --show_keypoints --match_threshold 0.2 --keypoint_threshold 0.03 --no_display --output_dir dump_match_pairs/20241121/ --input Ruun_code/back.avi

python3 20241121_RuunPose-realtime.py --anchor assets/Ruun_images/viewpoint/anchor/realAnchor.png --resize 1280 720 --superglue outdoor --nms_radius 3 --show_keypoints --match_threshold 0.2 --keypoint_threshold 0.03 --no_display --input Ruun_code/back.avi

python3 20241121_RuunPose-realtime.py --anchor assets/Ruun_images/viewpoint/anchor/realAnchor.png --resize 1280 720 --superglue outdoor --nms_radius 3 --show_keypoints --match_threshold 0.2 --keypoint_threshold 0.03 --no_display --input Ruun_code/rotate.avi

# 20241122
python3 20241121_RuunPose-realtime.py --anchor assets/Ruun_images/viewpoint/anchor/realAnchor.png --resize 1280 720 --superglue outdoor --nms_radius 3 --show_keypoints --match_threshold 0.2 --keypoint_threshold 0.03 --no_display --input assets/Ruun_images/video/20241122/ --output_dir dump_match_pairs/20241122/1/

python3 20241121_RuunPose-realtime.py --anchor assets/Ruun_images/viewpoint/anchor/realAnchor.png --resize 1280 720 --superglue outdoor --nms_radius 3 --show_keypoints --match_threshold 0.2 --keypoint_threshold 0.03 --input assets/Ruun_images/video/20241122/20241122_outdoor_sun_2.avi --no_display --output_dir dump_match_pairs/20241122/1/

python3 20241121_RuunPose-realtime.py --anchor assets/Ruun_images/viewpoint/anchor/realAnchor.png --resize 1280 720 --superglue outdoor --nms_radius 3 --show_keypoints --match_threshold 0.2 --keypoint_threshold 0.03 --input assets/Ruun_images/video/20241122/20241122_outdoor_sun_2.avi --no_display

python3 20241121_RuunPose-realtime.py --anchor assets/Ruun_images/viewpoint/anchor/realAnchor.png --resize 1280 720 --superglue outdoor --nms_radius 3 --show_keypoints --match_threshold 0.2 --keypoint_threshold 0.03 --input assets/Ruun_images/video/20241122/20241122_outdoor_sun_2.avi --no_display --output_dir dump_match_pairs/2024112

#20241128
python3 20241128_RTK_POSE.py --anchor assets/Ruun_images/viewpoint/anchor/realAnchor.png --resize 1280 720 --superglue outdoor --nms_radius 3 --show_keypoints --match_threshold 0.2 --keypoint_threshold 0.03 --input assets/Ruun_images/video/20241128/20241128_test.mp4 --no_display --output_dir dump_match_pairs/20241128/5/

python3 20241128_RTK.py --input assets/Ruun_images/video/20241128/20241128_test.mp4 --anchor assets/Ruun_images/viewpoint/anchor/realAnchor.png --output_timestamps frame_timestamps.csv --enu_file assets/RTK/20241128/last/reach_rover_solution_20241128101701.LLH 

#20241129

python3 20241129_RuunPose_KF_NoiseHandle.py --anchor assets/Ruun_images/viewpoint/anchor/realAnchor.png --resize 1280 720 --superglue outdoor --nms_radius 3 --show_keypoints --match_threshold 0.2 --keypoint_threshold 0.03 --input assets/Ruun_images/video/20241129/vibration_test.mp4 --no_display --output_dir dump_match_pairs/20241129/?

python3 main.py --anchor assets/Ruun_images/viewpoint/anchor/realAnchor.png --resize 1280 720 --superglue outdoor --nms_radius 3 --show_keypoints --match_threshold 0.2 --keypoint_threshold 0.03 --input assets/Ruun_images/video/20241129/vibration_test.mp4 --no_display --output_dir dump_match_pairs/20241129/?

python3 main.py --anchor assets/Ruun_images/viewpoint/anchor/realAnchor.png --resize 1280 720 --superglue outdoor --nms_radius 3 --show_keypoints --match_threshold 0.2 --keypoint_threshold 0.03 --input assets/Ruun_images/video/20241129/Big.mp4 --no_display --output_dir dump_match_pairs/20241129/?

python3 main.py --anchor assets/Ruun_images/viewpoint/anchor/realAnchor.png --superglue outdoor --nms_radius 3 --match_threshold 0.2 --keypoint_threshold 0.03 --input assets/Ruun_images/video/20241129/viewpoint.mp4 --no_display --resize 1280 720

python3 main_fast.py --anchor assets/Ruun_images/viewpoint/anchor/realAnchor.png --superglue outdoor --nms_radius 3 --match_threshold 0.2 --keypoint_threshold 0.03 --input assets/Ruun_images/video/20241129/viewpoint.mp4 --no_display --resize 1280 720

python3 main.py --anchor assets/Ruun_images/viewpoint/anchor/realAnchor_2.png --superglue outdoor --nms_radius 3 --match_threshold 0.2 --keypoint_threshold 0.03 --input assets/Ruun_images/video/20241129/C0083.MP4 --no_display --resize 1920 1080 --output_dir dump_match_pairs/20241203/3/

#20241204
python3 main.py --anchor assets/Ruun_images/viewpoint/anchor/Proto_under.png --superglue outdoor --nms_radius 3 --match_threshold 0.2 --keypoint_threshold 0.003 --input assets/Ruun_images/video/20241204/20240628_fly3.mp4 --no_display --resize 1280 720 --output_dir dump_match_pairs/20241203/3/ --show_keypoints

./match_pairs.py --resize 1600 --superglue outdoor --max_keypoints 2048 --nms_radius 3  --resize_float --input_dir dump_match_pairs/20241204/2/ --input_pairs assets/scannet_sample_pairs_with_gt.txt --output_dir dump_match_pairs/20241204/3/ --viz

#20241206
python3 main.py --anchor assets/Ruun_images/viewpoint/anchor/realAnchor.png --superglue outdoor --nms_radius 3 --match_threshold 0.2 --keypoint_threshold 0.03 --input assets/Ruun_images/video/20241129/viewpoint.mp4 --no_display --resize 1280 720 --output_dir dump_match_pairs/20241206/1/

#20241210
python3 main.py --anchor assets/Ruun_images/viewpoint/anchor/realAnchor.png --superglue outdoor --nms_radius 3 --match_threshold 0.2 --keypoint_threshold 0.03 --input assets/Ruun_images/video/20241210/Steady.mp4 --no_display --resize 1280 720 

python3 match_pairs.py --resize 1280 720 --superglue outdoor --max_keypoints 2048 --nms_radius 3  --resize_float --input_dir dump_match_pairs/20241210/test/ --input_pairs assets/scannet_sample_pairs_with_gt.txt --output_dir dump_match_pairs/20241210/match/ --show_keypoints --match_threshold 0.2 --keypoint_threshold 0.03 --viz

#20241212
python3 main_LG.py --anchor assets/Ruun_images/viewpoint/anchor/realAnchor.png --input assets/Ruun_images/video/20241210/Steady.mp4 --resize 1280 720

#20241213
python3 main_LG.py --anchor assets/Ruun_images/viewpoint/anchor/realAnchor.png --input assets/Ruun_images/video/20241210/Steady.mp4 --resize 1280 720

python3 main.py --anchor assets/Ruun_images/viewpoint/anchor/realAnchor.png --input assets/Ruun_images/video/20241210/Steady.mp4 --resize 1280 720 --show_keypoints --superglue outdoor

python3 main.py --anchor assets/Ruun_images/viewpoint/anchor/realAnchor.png --input assets/Ruun_images/video/20241213/WideBaseLine.mp4 --resize 1280 720 --no_display --output_dir dump_match_pairs/20241213/WideBase/ --superglue indoor

python3 main.py --anchor assets/Ruun_images/viewpoint/anchor/realAnchor.png --input assets/Ruun_images/video/20241213/WideBaseLine.mp4 --resize 1280 720 --no_display --output_dir dump_match_pairs/20241213/WideBase/ --superglue outdoor

python3 main.py --anchor assets/Ruun_images/viewpoint/anchor/realAnchor.png --input assets/Ruun_images/video/20241213/Difficult.mp4 --resize 1280 720 --no_display --superglue outdoor

python3 main.py --anchor assets/Ruun_images/viewpoint/anchor/realAnchor.png --input assets/Ruun_images/video/20241213/Big.mp4 --resize 1280 720 --no_display --superglue outdoor

python3 main.py --anchor assets/Ruun_images/viewpoint/anchor/realAnchor.png --input assets/Ruun_images/video/20241213/Roll_rotation.mp4 --resize 1280 720 --no_display --superglue outdoor --output_dir dump_match_pairs/20241213/Rotate/

python3 main_LG.py --anchor assets/Ruun_images/viewpoint/anchor/realAnchor.png --input assets/Ruun_images/video/20241210/Steady.mp4 --resize 1280 720

#20241219

python3 main_LG_Box_GT.py --anchor assets/Ruun_images/video/20241218/Box_Anchor/Opti_Box_Anchor.png --input assets/Ruun_images/video/20241218/20241218_HD.mp4 --resize 1280 720

python3 main_LG_GT_frame.py --anchor assets/Ruun_images/viewpoint/anchor/realAnchor.png --input assets/Ruun_images/video/20241217/Opti_Test1.mp4 --resize 1280 720

python3 Debug_29241219_GTvsMine

#290241224

python3 main_LG_GT_frame.py --anchor assets/Ruun_images/viewpoint/anchor/realAnchor.png --input assets/Ruun_images/video/20241224/20241224_new_cali.mp4 --resize 1280 720

python3 Debug_29241219_GTvsMine

python3 main_LG_GT_frame.py --anchor assets/Ruun_images/viewpoint/anchor/realAnchor.png --input assets/Ruun_images/video/20241224/20241224_test2.mp4 --resize 1280 720

#20241226

python3 main_formation_aircraft.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --input assets/Ruun_images/video/20241226/20241226_test3.mp4 --resize 1280 720

#20241227

python3 main_formation_aircraft.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --input assets/Ruun_images/video/20241227/20241227_test1.mp4 --resize 1280 720 --save_pose 20241227_test1.json

python3 main_formation_aircraft.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --input assets/Ruun_images/video/20241227/20241227_test1.mp4 --resize 1280 720 --save_pose 20241227_C_test1_anlaysis.json --output_dir dump_match_pairs/20241229/

#20241229

python3 main_formation_aircraft.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --input assets/Ruun_images/video/20241227/20241227_test1.mp4 --resize 1280 720 --save_pose 20241227_C_test1_anlaysis.json --output_dir dump_match_pairs/20241229/


python3 main_formation_aircraft.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --input assets/Ruun_images/video/20241227/20241227_test1.mp4 --resize 1280 720 --save_pose 20241227_C_test1_anlaysis_KF_10hz.json --output_dir dump_match_pairs/20241229/test1_KF_10hz/

python3 main_formation_aircraft.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor_side.png --input assets/Ruun_images/video/20241227/20241227_test1.mp4 --resize 1280 720 --save_pose 20241227_C_test1_new_anchor_kd.json --output_dir dump_match_pairs/20241229/test1_new_anchor_kd/

#20241230

python3 main_formation_aircraft.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --input assets/Ruun_images/video/20241230/20241230_test1.mp4 --resize 1280 720 --save_pose 20241230_test1.json --output_dir dump_match_pairs/20241230/test1/

