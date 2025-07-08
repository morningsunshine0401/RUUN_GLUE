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

#20250106

python3 main_formation_aircraft.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --input assets/Ruun_images/video/20250106/20250106_test2.mp4 --resize 1280 720 --save_pose 20250106_test2.json --output_dir dump_match_pairs/20250106/


python3 main_formation_aircraft.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --input assets/Ruun_images/video/20250106/20250106_test3.mp4 --resize 1280 720 --save_pose 20250106_test3.json --output_dir dump_match_pairs/20250106/test3/

#20250107

python3 20250106_main.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --input assets/Ruun_images/video/20250106/20250106_test2.mp4 --resize 1280 720 --save_pose 20250107_test2.json --output_dir dump_match_pairs/20250107/

python3 main_formation_aircraft.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --input assets/Ruun_images/video/20250107/20250107_test1.mp4 --resize 1280 720 --save_pose 20250107_test1.json --output_dir dump_match_pairs/20250107/test1/

python3 20250106_main.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --input assets/Ruun_images/video/20250107/20250107_test2.mp4 --resize 1280 720 --save_pose 20250107_test2_thresh_15cm.json --output_dir dump_match_pairs/20250107/test2_thresh15/

#20250108

python3 20250106_main.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --input assets/Ruun_images/video/20250107/20250107_test2.mp4 --resize 1280 720 --save_pose 20250108_test2_thresh.json --output_dir dump_match_pairs/20250108/

python3 main_formation_aircraft.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --input assets/Ruun_images/video/20250107/20250107_test3.mp4 --resize 1280 720 --save_pose 20250108_thresh_test3.json --output_dir dump_match_pairs/20250108/test3/

python3 main_formation_aircraft.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --input assets/Ruun_images/video/20250107/20250107_test18.mp4 --resize 1280 720 --save_pose 20250108_thresh_test18.json --output_dir dump_match_pairs/20250108/test18/

python3 main_formation_aircraft.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --input assets/Ruun_images/video/20250107/20250107_test16.mp4 --resize 1280 720 --save_pose 20250108_thresh_test16.json --output_dir dump_match_pairs/20250108/test16/

python3 20250108_main_pnp.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --input assets/Ruun_images/video/20250107/20250107_test2.mp4 --resize 1280 720 --save_pose 20250108_test2_pnp.json --output_dir dump_match_pairs/20250108/pnp/

python3 main_formation_aircraft.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --input assets/Ruun_images/video/20250108/20250108_far.mp4 --resize 1280 720 --save_pose 20250108_far.json --output_dir dump_match_pairs/20250108/far/

#20250109

python3 20250108_main_pnp.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --input assets/Ruun_images/video/20250107/20250107_test2.mp4 --resize 640 360 --save_pose 20250108_test2_resol_D.json --output_dir dump_match_pairs/20250109/

#20250115

python3 main_ICUAS.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --input assets/Ruun_images/video/20250107/20250107_test2.mp4 --resize 1280 720 --save_pose 20250115_test2_ICUAS.json --output_dir dump_match_pairs/20250115/

python3 main_ORB_ICUAS.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --input assets/Ruun_images/video/20250107/20250107_test2.mp4 --resize 1280 720 --save_pose 20250115_test2_ORB_ICUAS.json --output_dir dump_match_pairs/20250115/ORB/

python3 main_ORB_ICUAS.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --input assets/Ruun_images/video/20250107/20250107_test2.mp4 --resize 1280 720 --save_pose 20250115_test2_ORB_ICUAS.json --no_display

#20250120

python3 main_sync.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --resize 1280 720 --save_pose 20250121_test5.json --image_dir assets/Ruun_images/ICUAS/test5/ --csv_file assets/Ruun_images/ICUAS/test5/image_index.csv 

#20250121

python3 main_sync.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --resize 1280 720 --save_pose 20250121_test5.json --image_dir assets/Ruun_images/ICUAS/test5/ --csv_file assets/Ruun_images/ICUAS/test5/image_index.csv 

#20250122

python3 main_20250122_sync.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --resize 1280 720 --save_pose 20250122_test6.json --image_dir assets/Ruun_images/ICUAS/test6/ --csv_file assets/Ruun_images/ICUAS/test6/image_index.csv 

python3 main_sync.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --resize 1280 720 --save_pose 20250122_test5_adaptive.json --image_dir assets/Ruun_images/ICUAS/test5/ --csv_file assets/Ruun_images/ICUAS/test5/image_index.csv --no_display


python3 main_sync.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --resize 1280 720 --save_pose 20250122_opti_test1.json --image_dir assets/Ruun_images/ICUAS/20250122/extracted_images_20250122_test1/ --csv_file assets/Ruun_images/ICUAS/20250122/extracted_images_20250122_test1/image_index.csv --no_display


#20250123

# 133 frame
python3 main_sync.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --resize 1280 720 --save_pose 20250123_ICUAS_1.json --image_dir assets/Ruun_images/ICUAS/20250121/extracted_images_test5/ --csv_file assets/Ruun_images/ICUAS/20250121/extracted_images_test5/image_index.csv 


python3 main_sync.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --resize 1280 720 --save_pose 20250123_ICUAS_2_dark.json --image_dir assets/Ruun_images/ICUAS/20250122/extracted_images_test8/ --csv_file assets/Ruun_images/ICUAS/20250122/extracted_images_20250122_test8/image_index.csv --output_dir dump_match_pairs/20250123/dark/

#20250124

# 152 or 135 frame
python3 main_sync.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --resize 1280 720 --save_pose 20250124_ICUAS1.json --image_dir assets/Ruun_images/ICUAS/20250124/extracted_images_20250124_ICUAS1/ --csv_file assets/Ruun_images/ICUAS/20250124/extracted_images_20250124_ICUAS1/image_index.csv --output_dir dump_match_pairs/20250124/

# 169 or 144 frame
python3 main_sync.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --resize 1280 720 --save_pose 20250124_ICUAS1_2.json --image_dir assets/Ruun_images/ICUAS/20250124/c_ICUAS1_2/ --csv_file assets/Ruun_images/ICUAS/20250124/extracted_images_20250124_ICUAS1_2/image_index.csv --output_dir dump_match_pairs/20250124/ICUAS2/ --no_display

python3 main_sync.py --anchor Anchor_B.png --resize 1280 720 --save_pose boundary.json --image_dir assets/Ruun_images/ICUAS/20250124/Boundary/ --csv_file assets/Ruun_images/ICUAS/20250124/Boundary/image_index.csv --output_dir dump_match_pairs/20250124/boundary/

python3 main_sync.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --resize 1280 720 --save_pose boundary_front.json --image_dir assets/Ruun_images/ICUAS/20250124/Boundary/ --csv_file assets/Ruun_images/ICUAS/20250124/Boundary/image_index.csv --output_dir dump_match_pairs/20250124/boundary_front/

python3 main_ORB_ICUAS.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --resize 1280 720 --save_pose ORB_dark.json --input assets/Ruun_images/ICUAS/20250122/extracted_images_20250122_test8/ --output_dir dump_match_pairs/20250124/ORB_dark/

python3 main_sync.py  --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --resize 1280 720 --save_pose ORB_sync_dark.json --image_dir assets/Ruun_images/ICUAS/20250122/extracted_images_20250122_test8/ --output_dir dump_match_pairs/20250124/ORB_dark/ --csv_file assets/Ruun_images/ICUAS/20250122/extracted_images_20250122_test8/image_index.csv

python3 main_ICUAS.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --input assets/Ruun_images/video/20250107/20250107_test2.mp4 --resize 1280 720 --save_pose coverage_0.25.json 

#20250203

python3 main_sync.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --resize 1280 720 --save_pose 20250123_ICUAS_1.json --image_dir assets/Ruun_images/ICUAS/20250203/extracted_images_20250203_test1/ --csv_file assets/Ruun_images/ICUAS/2025203/extracted_images_20250203_test1/image_index.csv 

#20250204

python3 main_sync.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --resize 1280 720 --image_dir assets/Ruun_images/ICUAS/20250203/extracted_images_20250203_test1/ --csv_file assets/Ruun_images/ICUAS/20250203/extracted_images_20250203_test1/image_index.csv --save_pose 20250204_ICUAS_remove_test.json --output_dir dump_match_pairs/20250204/

#20250210

python3 main_sync.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --resize 1280 720 --image_dir assets/Ruun_images/ICUAS/20250128/extracted_images_20250128_test4/ --csv_file assets/Ruun_images/ICUAS/20250128/extracted_images_20250128_test4/image_index.csv --save_pose 20250210_test2_KF_anlaysis_10hz.json


python3 main_sync.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --resize 1280 720 --image_dir assets/Ruun_images/ICUAS/20250210/extracted_images_test5/ --csv_file assets/Ruun_images/ICUAS/20250210/extracted_images_test5/image_index.csv --save_pose 20250210_test5_KF_anlaysis_10hz.json


python3 main_sync.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --resize 1280 720 --image_dir assets/Ruun_images/ICUAS/20250210/extracted_images_test5/ --csv_file assets/Ruun_images/ICUAS/20250210/extracted_images_test5/image_index.csv --save_pose 20250210_test5_KF_anlaysis_30hz.json


python3 main_sync.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --resize 1280 720 --image_dir assets/Ruun_images/ICUAS/20250128/extracted_images_20250128_test4/ --csv_file assets/Ruun_images/ICUAS/20250128/extracted_images_20250128_test4/image_index.csv --save_pose 20250210_test4_adaptive_harsh.json

python3 main_sync.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --resize 1280 720 --image_dir assets/Ruun_images/ICUAS/20250210/extracted_images_test5/ --csv_file assets/Ruun_images/ICUAS/20250210/extracted_images_test5/image_index.csv --save_pose 20250210_test5_adaptive2.json 

python3 main_sync.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --resize 1280 720 --image_dir assets/Ruun_images/ICUAS/20250210/extracted_images_test5/ --csv_file assets/Ruun_images/ICUAS/20250210/extracted_images_test5/image_index.csv --save_pose 20250212_test5_adaptive2.json 

python3 main_sync.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --resize 1280 720 --image_dir assets/Ruun_images/ICUAS/20250128/extracted_images_20250128_test4/ --csv_file assets/Ruun_images/ICUAS/20250128/extracted_images_20250128_test4/image_index.csv --save_pose 20250212_test4_adaptive2.json 

# 20250217 Kalman filter analysis with steady


python3 main_sync.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --resize 1280 720 --save_pose 20250217_20250203_test2_steady.json --image_dir assets/Ruun_images/ICUAS/20250203/extracted_images_20250203_test2/ --csv_file assets/Ruun_images/ICUAS/2025203/extracted_images_20250203_test2/image_index.csv --output_dir dump_match_pairs/20250217/

python3 main_sync.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --resize 1280 720 --image_dir assets/Ruun_images/ICUAS/20250203/extracted_images_20250203_test2/ --csv_file assets/Ruun_images/ICUAS/20250203/extracted_images_20250203_test2/image_index.csv --save_pose 20250217_20250203_test2_steady_measurementNoiseCov_increase_processNoise_decrease.json --no_display

python3 main_sync.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --resize 1280 720 --image_dir assets/Ruun_images/ICUAS/20250128/extracted_images_20250128_test4/ --csv_file assets/Ruun_images/ICUAS/20250128/extracted_images_20250128_test4/image_index.csv --no_display --save_pose 20250217_20250128_test4_kalmanfilter_tuning.json 


# 20250218 4 cases for choosing the KF update logic

python3 main_sync.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --resize 1280 720 --image_dir assets/Ruun_images/ICUAS/20250128/extracted_images_20250128_test4/ --csv_file assets/Ruun_images/ICUAS/20250128/extracted_images_20250128_test4/image_index.csv --save_pose 20250218_20250128_test4_4cases.json --output_dir dump_match_pairs/20250218/


python3 main_sync.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --resize 1280 720 --image_dir assets/Ruun_images/ICUAS/20250128/extracted_images_20250128_test4/ --csv_file assets/Ruun_images/ICUAS/20250128/extracted_images_20250128_test4/image_index.csv --no_display --save_pose 20250218_20250128_test4_4cases.json 

python3 main_sync.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --resize 1280 720 --image_dir assets/Ruun_images/ICUAS/20250203/extracted_images_20250203_test2/ --csv_file assets/Ruun_images/ICUAS/20250203/extracted_images_20250203_test2/image_index.csv --save_pose 20250218_20250203_test2_steady_zero.json --no_display


# 20250219 matching quality and RANSAC tuning

python3 main_sync.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --resize 1280 720 --image_dir assets/Ruun_images/ICUAS/20250128/extracted_images_20250128_test4/ --csv_file assets/Ruun_images/ICUAS/20250128/extracted_images_20250128_test4/image_index.csv --save_pose 20250219_20250128_test4_matching.json --output_dir dump_match_pairs/20250219/

python3 main_sync.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --resize 1280 720 --image_dir assets/Ruun_images/ICUAS/20250203/extracted_images_20250203_test2/ --csv_file assets/Ruun_images/ICUAS/20250203/extracted_images_20250203_test2/image_index.csv --save_pose 20250219_20250203_test2_steady.json --no_display


# 20250224 Quaternion test 1

python3 main_sync.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --resize 1280 720 --image_dir assets/Ruun_images/ICUAS/20250128/extracted_images_20250128_test4/ --csv_file assets/Ruun_images/ICUAS/20250128/extracted_images_20250128_test4/image_index.csv --save_pose 20250219_20250128_test4_Q.json


# 20250304 

python3 main_sync.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --resize 1280 720 --image_dir assets/Ruun_images/ICUAS/20250128/extracted_images_20250128_test4/ --csv_file assets/Ruun_images/ICUAS/20250128/extracted_images_20250128_test4/image_index.csv --save_pose 20250304_20250128_test4_Q.json

python3 main_sync.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --resize 1280 720 --image_dir assets/Ruun_images/ICUAS/20250128/extracted_images_20250128_test4/ --csv_file assets/Ruun_images/ICUAS/20250128/extracted_images_20250128_test4/image_index.csv --save_pose 20250304_20250128_test4_pixel.json


# 20250305 Tested with Blender to checck for calibration effectiveness

python3 RuunPose-3D-viewpoint-blender.py --input assets/Ruun_images/viewpoint/test/rotated/ --ground_truth assets/Ruun_images/viewpoint/test/rotated/viewpoint_GT_rotate.json --resize 1280 960 --superglue outdoor --show_keypoints --viewpoint_model_path viewpoint_model_more_data.pth --output_dir dump_match_pairs/match_output/viewpoint/result/rotate/ --show_keypoints

python3 main_sync.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --resize 1280 960 --image_dir --input assets/Ruun_images/viewpoint/test/rotated/ --csv_file assets/Ruun_images/ICUAS/20250128/extracted_images_20250128_test4/image_index.csv --save_pose 20250304_20250128_test4_pixel_blender.json

python3 main_sync_blender.py --anchor assets/Ruun_images/viewpoint/anchor/70.png --resize 1280 960 --image_dir assets/Ruun_images/viewpoint/Blender/20250305/ --save_pose 20250305_pixel_blender.json --ground_truth assets/Ruun_images/viewpoint/Blender/20250305/viewpoint_GT_rotate.json 


# 20250306

python3 main_track.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --resize 1280 720 --image_dir assets/Ruun_images/ICUAS/20250128/extracted_images_20250128_test4/ --csv_file assets/Ruun_images/ICUAS/20250128/extracted_images_20250128_test4/image_index.csv --save_pose 20250304_20250128_test4_track.json


# 20250307

python3 main_track.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --resize 1280 720 --image_dir assets/Ruun_images/ICUAS/20250128/extracted_images_20250128_test4/ --csv_file assets/Ruun_images/ICUAS/20250128/extracted_images_20250128_test4/image_index.csv --save_pose 20250307_20250128_test4_track.json

python3 main_track.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --resize 1280 720 --image_dir assets/Ruun_images/ICUAS/20250203/extracted_images_20250203_test1/ --csv_file assets/Ruun_images/ICUAS/20250203/extracted_images_20250203_test1/image_index.csv --save_pose 20250307_20250203_test1_track.json



# 20250308

python3 main_thread.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --resize 1280 720 --image_dir assets/Ruun_images/ICUAS/20250128/extracted_images_20250128_test4/ --csv_file assets/Ruun_images/ICUAS/20250128/extracted_images_20250128_test4/image_index.csv --save_pose 20250308_20250128_test4_thread.json 

python3 main_sync.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --resize 1280 720 --image_dir assets/Ruun_images/ICUAS/20250128/extracted_images_20250128_test4/ --csv_file assets/Ruun_images/ICUAS/20250128/extracted_images_20250128_test4/image_index.csv --save_pose 20250308_20250128_test4_thread.json

python3 main_sync.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --resize 1280 720 --image_dir assets/Ruun_images/ICUAS/20250128/extracted_images_20250128_test4/ --csv_file assets/Ruun_images/ICUAS/20250128/extracted_images_20250128_test4/image_index.csv --save_pose 20250308_20250128_test4_two_model.json


# 20250310 Succeeded Threshing and trying out tracking

python3 main_thread_track.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --resize 1280 720 --image_dir assets/Ruun_images/ICUAS/20250128/extracted_images_20250128_test4/ --csv_file assets/Ruun_images/ICUAS/20250128/extracted_images_20250128_test4/image_index.csv --save_pose 20250310_20250128_test4_track.json

python3 main_thread_track.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --resize 1280 720 --image_dir assets/Ruun_images/ICUAS/20250203/extracted_images_20250203_test2/ --csv_file assets/Ruun_images/ICUAS/20250203/extracted_images_20250203_test2/image_index.csv --save_pose 202503010_20250203_test2_track.json

python3 main_track.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --resize 1280 720 --image_dir assets/Ruun_images/ICUAS/20250128/extracted_images_20250128_test4/ --csv_file assets/Ruun_images/ICUAS/20250128/extracted_images_20250128_test4/image_index.csv --save_pose 20250310_20250128_test4_track.json

# 20250311

python3 main_track.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --resize 1280 720 --image_dir assets/Ruun_images/ICUAS/20250210/extracted_images_test5/ --csv_file assets/Ruun_images/ICUAS/20250210/extracted_images_test5/image_index.csv --save_pose 20250311_20250210_test5_DIS_track.json 

python3 main_thread.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --resize 1280 720 --image_dir assets/Ruun_images/ICUAS/20250210/extracted_images_test5/ --csv_file assets/Ruun_images/ICUAS/20250210/extracted_images_test5/image_index.csv --save_pose 20250311_20250210_test5_thread.json 

python3 main_track.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --resize 1280 720 --image_dir assets/Ruun_images/ICUAS/20250203/extracted_images_20250203_test2/ --csv_file assets/Ruun_images/ICUAS/20250203/extracted_images_20250203_test2/image_index.csv --save_pose 202503011_20250203_test2_track_DIS.json

python3 main_thread.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --resize 1280 720 --image_dir assets/Ruun_images/ICUAS/20250203/extracted_images_20250203_test2/ --csv_file assets/Ruun_images/ICUAS/20250203/extracted_images_20250203_test2/image_index.csv --save_pose 202503011_20250203_test2_thread.json

python3 main_sync.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --resize 1280 720 --image_dir assets/Ruun_images/ICUAS/20250128/extracted_images_20250128_test4/ --csv_file assets/Ruun_images/ICUAS/20250128/extracted_images_20250128_test4/image_index.csv --save_pose 202503011_20250128_test4_pixel.json

python3 main_sync.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --resize 1280 720 --image_dir assets/Ruun_images/ICUAS/20250203/extracted_images_20250203_test2/ --csv_file assets/Ruun_images/ICUAS/20250203/extracted_images_20250203_test2/image_index.csv --save_pose 202503011_20250203_test2_pixel.json

python3 main_thread.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --resize 1280 720 --image_dir assets/Ruun_images/ICUAS/20250128/extracted_images_20250128_test4/ --csv_file assets/Ruun_images/ICUAS/20250128/extracted_images_20250128_test4/image_index.csv --save_pose 20250311_20250128_test4_thread.json

# 20250312

python3 main_thread.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --resize 1280 720 --image_dir assets/Ruun_images/ICUAS/20250128/extracted_images_20250128_test4/ --csv_file assets/Ruun_images/ICUAS/20250128/extracted_images_20250128_test4/image_index.csv --save_pose 20250312_20250128_test4_tight.json

python3 main_thread.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --resize 1280 720 --image_dir assets/Ruun_images/ICUAS/20250203/extracted_images_20250203_test2/ --csv_file assets/Ruun_images/ICUAS/20250203/extracted_images_20250203_test2/image_index.csv --save_pose 202503012_20250203_test2_tight.json

# 20250313 Webcam

python3 main_webcam.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --device cuda --resize 1280 720

python3 main_webcam.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --device cuda --resize 1280 720 --dual_resolution --process_width 640 --process_height 480 --camera_width 640 --camera_height 480



python3 main_webcam.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --device cuda --resize 1280 720 --camera_width 640 --camera_height 480

python3 main_webcam.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --device cuda --camera_width 640 --camera_height 480

python3 main_webcam.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --device cuda --camera_width 640 --camera_height 480

python3 main_webcam.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --device cuda --camera_width 640 --camera_height 480 --optimize --optimization_level mild

python3 main_webcam.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --device cuda --dual_resolution --process_width 640 --process_height 480 --camera_width 640 --camera_height 480

python3 main_thread.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --resize 1280 720 --image_dir assets/Ruun_images/ICUAS/20250128/extracted_images_20250128_test4/ --csv_file assets/Ruun_images/ICUAS/20250128/extracted_images_20250128_test4/image_index.csv --save_pose 20250313_20250128_test4_tight.json

python3 main_sync.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --resize 1280 720 --image_dir assets/Ruun_images/ICUAS/20250128/extracted_images_20250128_test4/ --csv_file assets/Ruun_images/ICUAS/20250128/extracted_images_20250128_test4/image_index.csv --save_pose 202503013_20250128_test4.json

python3 main_webcam.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --device cuda --camera_width 1280 --camera_height 720

python3 main_webcam.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --device cuda --camera_width 1280 --camera_height 720 --save_consolidated_json

# 20250314

python3 main_webcam.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --device cuda --camera_width 1280 --camera_height 720 --save_consolidated_json


# 20250317

python3 main_webcam.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --device cuda --camera_width 1280 --camera_height 720 --save_consolidated_json

CUDA_LAUNCH_BLOCKING=1 python3 main_webcam.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --device cuda --camera_width 1280 --camera_height 720 --save_consolidated_json


python3 main_webcam.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --device cuda --dual_resolution --process_width 1280 --process_height 720 --camera_width 1280 --camera_height 720 --save_consolidated_json

python3 main_webcam.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --device cuda --camera_width 1280 --camera_height 720 --save_consolidated_json


#20250319

python3 main_webcam.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --device cuda --camera_width 1280 --camera_height 720 --save_consolidated_json --KF_mode L

python3 main_webcam.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --device cuda --camera_width 640 --camera_height 480 --save_consolidated_json --KF_mode L --process_width 640 --process_height 480



#20250320 

python3 main_thread.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --resize 1280 720 --image_dir assets/Ruun_images/ICUAS/20250128/extracted_images_20250128_test4/ --csv_file assets/Ruun_images/ICUAS/20250128/extracted_images_20250128_test4/image_index.csv --save_pose 20250320_20250128_test4.json

python3 main_webcam.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --device cuda --camera_width 1280 --camera_height 720 --save_consolidated_json --KF_mode L


#20250324

python3 demo_superglue.py --input DJI_Clean.mp4 --anchor T1.png --superglue outdoor 


# 20250325

python3 main_webcam.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --device cuda --camera_width 1280 --camera_height 720 --save_consolidated_json --KF_mode L

python3 main_webcam.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --device cuda --camera_width 1280 --camera_height 720 --save_consolidated_json --KF_mode L --consolidated_json_filename Blur.json

python3 main_webcam.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --device cuda --camera_width 1280 --camera_height 720 --save_consolidated_json --KF_mode T --consolidated_json_filename T1.json

watch -n 1 sensors

python3 main_webcam.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --device cuda --camera_width 1280 --camera_height 720 --save_consolidated_json --KF_mode L --consolidated_json_filename KF_Upgrade.json


# 20250326


python3 main_webcam.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --device cuda --camera_width 640 --camera_height 480 --save_consolidated_json --KF_mode L --consolidated_json_filename KF_Upgrade.json


python3 main_webcam.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --device cuda --camera_width 1280 --camera_height 720 --save_consolidated_json --KF_mode L --consolidated_json_filename KF_Upgrade.json --dual_resolution --process_width 640 --process_height 480


python3 main_webcam.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --device cuda --camera_width 640 --camera_height 480 --save_consolidated_json --KF_mode T --consolidated_json_filename T1.json


# 20250331

python3 match_pairs.py --resize 1600 --superglue outdoor --max_keypoints 2048 --nms_radius 3  --resize_float --input_dir dump_match_pairs/20241204/2/ --input_pairs assets/scannet_sample_pairs_with_gt.txt --output_dir dump_match_pairs/20241204/3/ --viz



# 20250401

python3 demo_superglue.py --anchor 1000029745.png --input 20250331_real_video_2.mp4 --superglue outdoor --match_threshold 0.25



# 20250414

python3 demo_superglue_2d_info.py --anchor Back_CAD.png --input DJI_0122.MP4 --superglue outdoor --match_threshold 0.25 --output_dir dump_match_pairs/20250414/ --no_display


python3 20241218_manual_keypoint_on_image.py 


python3 demo_superglue_2d_info.py --anchor Back_CAD.png --input CAD/ --superglue outdoor --match_threshold 0.25 --output_dir dump_match_pairs/20250414/CAD/ --no_display --resize 742 672


python3 main_MK4.py --anchor Back_CAD.png --device cuda --input DJI_back.mp4 --save_consolidated_json --KF_mode L --consolidated_json_filename 20250414_TRANS_CAD.json --resize 1920 1080



python3 demo_superglue_2d_info.py --anchor Back_CAD.png --input DJI_back.mp4 --superglue outdoor --match_threshold 0.25 --output_dir dump_match_pairs/20250414/ANALYSIS/ --no_display



python3 main_MK4.py --anchor Back_CAD.png --device cuda --input DJI_back2.mp4 --save_consolidated_json --KF_mode L --consolidated_json_filename 20250414_TRANS_CAD.json --resize 1920 1080



python3 main_MK4.py --anchor Back_CAD.png --device cuda --input DJI_back2.mp4 --save_consolidated_json --KF_mode L --consolidated_json_filename 20250414_TRANS_CAD.json --resize 1140 1044


# 20250415

python3 main_MK4.py --anchor Back_CAD.png --device cuda --input DJI_back.mp4 --save_consolidated_json --KF_mode L --consolidated_json_filename 20250415_TC_show_p.json --resize 1920 1080 --show_keypoints


python3 demo_superglue_2d_info.py --anchor Back_CAD.png --input DJI_back.mp4 --superglue outdoor --match_threshold 0.25 --output_dir dump_match_pairs/20250414/ANALYSIS/ --show_keypoints


python3 main_MK4.py --anchor Back_CAD.png --device cuda --input DJI_back.mp4 --save_consolidated_json --KF_mode L --consolidated_json_filename 20250415_TC_diff_intrin.json --resize 1920 1080 



# 20250416

python3 main_MK4.py --anchor Back_CAD.png --device cuda --input DJI_back2.mp4 --save_consolidated_json --KF_mode L --consolidated_json_filename 20250416.json --resize 1920 1080 


python3 demo_superglue_2d_info.py --anchor dust3r_3.png --input DJI_back.mp4 --superglue outdoor --match_threshold 0.25 --output_dir dump_match_pairs/20250416 --show_keypoints


python3 demo_superglue_2d_info.py --anchor Real_Back.png --input DJI_back.mp4 --superglue outdoor --match_threshold 0.25 --output_dir dump_match_pairs/20250416/real --show_keypoints --resize 1280 720


python3 20241218_manual_keypoint_on_image.py 


python3 main_MK4.py --anchor Real_Back.png --device cuda --input DJI_back.mp4 --save_consolidated_json --KF_mode L --consolidated_json_filename 20250416_real.json --resize 1280 720 

python3 main_MK4.py --anchor Real_Back.png --device cuda --input DJI_back2.mp4 --save_consolidated_json --KF_mode L --consolidated_json_filename 20250416_real.json --resize 570 522 


# 20250421

python3 main_MK4.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --device cuda --input far_0421.mp4 --save_consolidated_json --KF_mode L --consolidated_json_filename 20250421_far.json --resize 1280 720 



python3 demo_superglue_2d_info.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --input far_0421.mp4 --superglue outdoor --match_threshold 0.25 --output_dir dump_match_pairs/20250421/ --show_keypoints --resize 1280 720


# 20250423

# Basic usage - measurements every 5 steps with moderate noise
python3 kf_analysis_tool.py --meas-interval 5 --noise-pos 0.05 --noise-quat 0.01

# Track Kalman gains and covariances for detailed analysis
python3 kf_analysis_tool.py --meas-interval 3 --noise-pos 0.1 --track-gains

# Save plots to files
python3 kf_analysis_tool.py --meas-interval 10 --noise-pos 0.2 --track-gains --save-plots


python3 kf_analysis_tool.py --meas-interval 5 --noise-pos 0.05 --noise-quat 0.01 --track-gains


python3 kf_analysis_tool.py --simple-trajectory --meas-interval 10 --noise-pos 0.05



python3 kf_analysis_tool.py --meas-interval 3 --noise-pos 0.05 --noise-quat 0.01


python3 kf_analysis_tool.py --simple-trajectory --meas-interval 4 --noise-pos 0.05 --noise-quat 0.01


python3 kf_debug_runner.py --update-mod 2 --update-phase 0


python3 kf_analysis_tool.py --meas-interval 3 --noise-pos 0.05 --noise-quat 0.01 --track-gains



python3 kf_analysis_tool.py --meas-interval 3 --noise-pos 0.05 --noise-quat 0.01 --track-gains --save-plots



# Process model check


python3 kf_process_check.py --warmup 3 --noise-pos 0.01


python3 kf_process_check.py --warmup 3 --noise-pos 0.03 --noise-quat 0.02



# 20250502


python3 demo_superglue_2d_info.py --anchor Heechan.png --input Heechan_20250502.mp4 --superglue outdoor --match_threshold 0.25 --output_dir dump_match_pairs/20250502/ --show_keypoints --resize 1280 720


python3 demo_superglue_2d_info.py --anchor Heechan.png --input 20250502_far1_1.mp4 --superglue outdoor --match_threshold 0.25 --output_dir dump_match_pairs/20250502/far1/ --show_keypoints --resize 1280 720


python3 main_MK4.py --anchor Real_Back.png --device cuda --input DJI_back.mp4 --save_consolidated_json --KF_mode L --consolidated_json_filename 20250502_real.json --resize 1280 720 


python3 main_MK4.py --anchor Real_Back.png --device cuda --input DJI_back2.mp4 --save_consolidated_json --KF_mode L --consolidated_json_filename 20250502_real2.json --resize 570 522 


20250331_real_video_2


python3 main_MK4.py --anchor Real_Back.png --device cuda --input 20250502_real_video_1.mp4 --save_consolidated_json --KF_mode L --consolidated_json_filename 20250502_real.json --resize 950 522 #1280 720 


python3 demo_superglue_2d_info.py --anchor Real_Back.png --input 20250502_real_video_1.mp4 --superglue outdoor --match_threshold 0.25 --output_dir dump_match_pairs/20250502/real/ --show_keypoints --resize 950 522


python3 demo_superglue_2d_info.py --anchor Real_Back.png --input 20250331_real_video_2.mp4 --superglue outdoor --match_threshold 0.25 --output_dir dump_match_pairs/20250502/real/far/ --show_keypoints --resize 1920 1080


python3 main_MK4.py --anchor Real_Back.png --device cuda --input 20250331_real_video_2.mp4 --save_consolidated_json --KF_mode L --consolidated_json_filename 20250502_real_hard.json --resize 1920 1080 


python3 main_MK4.py --anchor Real_Back.png --device cuda --input 20250502_real_video_2.mp4 --save_consolidated_json --KF_mode L --consolidated_json_filename 20250502_real_3.json --resize 982 514

python3 main_MK4.py --anchor Real_Back.png --device cuda --input 20250502_real_video_2.mp4 --save_consolidated_json --KF_mode L --consolidated_json_filename 20250502_real_g.json --resize 982 514

python3 main_MK4.py --anchor Real_Back.png --device cuda --input 20250331_real_video_3.mp4 --save_consolidated_json --KF_mode L --consolidated_json_filename 20250502_real_g2.json --resize 1482 830



# 20250506

python3 main_MK4.py --anchor Real_Back.png --device cuda --input 20250331_real_video_3.mp4 --save_consolidated_json --KF_mode L --consolidated_json_filename 20250506_real.json --resize 1482 830

python3 main_MK4.py --anchor Real_Back.png --device cuda --input 20250502_real_video_2.mp4 --save_consolidated_json --KF_mode L --consolidated_json_filename 20250506_real_2.json --resize 982 514


python3 demo_superglue_2d_info.py --anchor Heechan.png --input 20250506_far.MP4 --superglue outdoor --match_threshold 0.25 --output_dir dump_match_pairs/20250506/ --show_keypoints --resize 1920 1080


python3 demo_superglue_2d_info.py --anchor Doll.png --input Doll.MP4 --superglue outdoor --match_threshold 0.25 --output_dir dump_match_pairs/20250506/Doll/ --show_keypoints --resize 1920 1080


python3 20241218_manual_keypoint_on_image.py 

python3 demo_superglue_2d_info.py --anchor Heechan.png --input 20250506_far.MP4 --superglue outdoor --match_threshold 0.25 --output_dir dump_match_pairs/20250506/ --show_keypoints --resize 1030 543


python3 demo_superglue_2d_info.py --anchor Heechan.png --input 20250506_far2.mp4 --superglue outdoor --match_threshold 0.25 --output_dir dump_match_pairs/20250506/far2/ --show_keypoints --resize 1280 720

python3 demo_superglue_2d_info.py --anchor Heechan.png --input 20250506_occlusion.mp4 --superglue outdoor --match_threshold 0.25 --output_dir dump_match_pairs/20250506/occlusion/ --show_keypoints --resize 1280 720

python3 demo_superglue_2d_info.py --anchor Heechan2.png --input 20250506_far2.mp4 --superglue outdoor --match_threshold 0.25 --resize 1280 720

python3 demo_superglue_2d_info.py --anchor Heechan.png --input 20250506_far2.mp4 --superglue outdoor --match_threshold 0.25 --resize 1030 543

python3 demo_superglue_2d_info.py --anchor Heechan3.png --input 20250506_far2.mp4 --superglue outdoor --match_threshold 0.25 --resize 1280 720

python3 demo_superglue_2d_info.py --anchor Heechan3.png --input 20250506_far3.mp4 --superglue outdoor --match_threshold 0.25 --resize 1280 720

python3 demo_superglue_2d_info.py --anchor Heechan.png --input 20250506_far.MP4 --superglue outdoor --match_threshold 0.25 --output_dir dump_match_pairs/20250506/ --show_keypoints --resize 1030 543

python3 demo_superglue_2d_info.py --anchor Heechan.png --input 20250506_far.MP4 --superglue outdoor --match_threshold 0.25 --show_keypoints --resize 1920 1080

python3 demo_superglue_2d_info.py --anchor Heechan3.png --input 20250506_far.MP4 --superglue outdoor --match_threshold 0.25 --show_keypoints --resize 1920 1080

python3 demo_superglue_2d_info.py --anchor Heechan3.png --input 20250506_far_short.mp4 --superglue outdoor --match_threshold 0.25 --show_keypoints --resize 1920 1080 --output_dir dump_match_pairs/20250506/far_short/ --no_display


python3 main_MK4.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --device cuda --input 20250506_test1.mp4 --save_consolidated_json --KF_mode L --consolidated_json_filename 20250506_test1.json --resize 1280 720 



# 20250507

python3 main_MK4.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --device cuda --input 20250507_test1.mp4 --save_consolidated_json --KF_mode L --consolidated_json_filename 20250507_test1.json --resize 1280 720 


python3 demo_superglue_2d_info.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --input 20250507_test1.mp4 --superglue outdoor --match_threshold 0.25 --show_keypoints --resize 1280 720



python3 demo_superglue_2d_info.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --input 20250507_test2.mp4 --superglue outdoor --match_threshold 0.25 --show_keypoints --resize 640 480


python3 main_MK4.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --device cuda --input 20250507_test1.mp4 --save_consolidated_json --KF_mode L --consolidated_json_filename 20250507_test1.json --resize 1280 720 


# 20250508

python3 main_thread.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --resize 1280 720 --image_dir assets/Ruun_images/ICUAS/20250128/extracted_images_20250128_test4/ --csv_file assets/Ruun_images/ICUAS/20250128/extracted_images_20250128_test4/image_index.csv --save_pose 20250508_20250128_test4.json


python3 main_sync.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --resize 1280 720 --image_dir assets/Ruun_images/ICUAS/20250128/extracted_images_20250128_test2/ --csv_file assets/Ruun_images/ICUAS/20250128/extracted_images_20250128_test2/image_index.csv --save_pose 20250508_20250128_test2.json


python3 main_MK4.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --device cuda --input 20250508_far.mp4 --save_consolidated_json --KF_mode L --consolidated_json_filename 20250508_far.json --resize 1280 720 


python3 main_MK4.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --device cuda --input 20250508_yaw.mp4 --save_consolidated_json --KF_mode L --consolidated_json_filename 20250508_4.json --resize 1280 720 



python3 main_sync.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --resize 1280 720 --image_dir assets/Ruun_images/ICUAS/20250203/extracted_images_20250203_test2/ --csv_file assets/Ruun_images/ICUAS/20250203/extracted_images_20250203_test2/image_index.csv --save_pose 20250508_20250203_test2_pixel.json


python3 main_MK4.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --device cuda --input 20250508_standard.mp4 --save_consolidated_json --KF_mode L --consolidated_json_filename 20250508_standard.json --resize 1280 720 

python3 main_MK4.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --device cuda --input 20250508_blur.mp4 --save_consolidated_json --KF_mode L --consolidated_json_filename 20250508_2.json --resize 1280 720 

python3 main_MK4.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --device cuda --input 20250508_edge.mp4 --save_consolidated_json --KF_mode L --consolidated_json_filename 20250508_3.json --resize 1280 720 

python3 main_MK4.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --device cuda --input 20250508_yaw.mp4 --save_consolidated_json --KF_mode L --consolidated_json_filename 20250508_4.json --resize 1280 720 



python3 main_MK4.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --device cuda --input 20250508_test2.mp4 --save_consolidated_json --KF_mode L --consolidated_json_filename 20250508_4.json --resize 1280 720 


python3 main_MK4.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --device cuda --input 20250507_test1.mp4 --save_consolidated_json --KF_mode L --consolidated_json_filename 20250508_t1.json --resize 1280 720 



python3 main_MK4.py --anchor Real_Back.png --device cuda --input 20250502_real_video_2.mp4 --save_consolidated_json --KF_mode L --consolidated_json_filename 20250506_real_2.json --resize 982 514


python3 main_MK4.py --anchor Real_Back.png --device cuda --input 20250502_real_video_2.mp4 --save_consolidated_json --KF_mode L --consolidated_json_filename 20250506_real_2.json --resize 1280 720



python3 main_MK4.py --anchor Real_Back.png --device cuda --input 20250502_real_video_1.mp4 --save_consolidated_json --KF_mode L --consolidated_json_filename 20250506_real_2.json --resize 950 522


python3 main_MK4.py --anchor Real_Back.png --device cuda --input 20250331_real_video_3.mp4 --save_consolidated_json --KF_mode L --consolidated_json_filename 20250506_real.json --resize 1482 830


# 20250512

python3 main_MK4.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --device cuda --input 20250508_blur.mp4 --save_consolidated_json --KF_mode L --consolidated_json_filename 20250512_blur.json --resize 1280 720 

python3 main_MK4.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --device cuda --input 20250508_edge.mp4 --save_consolidated_json --KF_mode L --consolidated_json_filename 20250512_edge.json --resize 1280 720 

python3 main_MK4.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --device cuda --input 20250508_yaw.mp4 --save_consolidated_json --KF_mode L --consolidated_json_filename 20250512_yaw.json --resize 1280 720 




# 20250522


python3 demo_superglue_2d_info.py --anchor LM_cat_f.png --input assets/LM/train/ --superglue outdoor --match_threshold 0.25 --output_dir dump_match_pairs/20250522/

python3 main_MK42.py --device cuda --camera_id 0 --camera_width 1280 --camera_height 720 --resize 1280 720 --KF_mode L --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png

python3 main_MK42.py --device cuda --camera_id 0 --camera_width 1280 --camera_height 720 --resize 1280 720 --KF_mode L --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png

python3 main_MK42.py --device cuda --camera_id 0 --camera_width 1280 --camera_height 720 --resize 1280 720 --KF_mode L --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png




# Generate full visual report
python3 aircraft_annotator.py --reference assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --input-folder annotation_input/ --visualize

# Then open: annotations/visualizations/report.html


# 20250605

python3 annotator_MK2.py --reference-folder references/ --num-keypoints 6 --input-folder annotation_input1/

python3 auto_annotator_DINOv2.py     --reference-config interactive_reference_config.json     --input-folder annotation_input1/     --object-name aircraft     --visualize --selected-method enhanced_hybrid --debug

python3 AutoAnnotate.py     --reference-config interactive_reference_config.json     --input-folder annotation_input1/     --object-name aircraft     --visualize --selection-method best_keypoint_match

python3 annotator_config.py --reference-folder references/ --num-keypoints 6

python3 AutoMK2.py     --reference-config interactive_reference_config.json     --input-folder annotation_input1/     --visualize 
# This one at the top of this line is good

python3 LLMMK1.py --reference-config interactive_reference_config.json --input-folder annotation_input1/ --object-name "white aircraft toy"






#20250609

python3 annotator_config.py --reference-folder references/coffee/ --num-keypoints 10

python3 AutoMK3.py     --reference-config interactive_reference_config.json     --input-folder annotation_input1/coffee/     --visualize 




# 20250615

python3 annotator_config.py --reference-folder references/ --num-keypoints 6

python3 AutoMK3.py     --reference-config interactive_reference_config.json     --input-folder annotation_input1/     --visualize 



python3 main_MK43.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --device cuda


# 20250616

python3 VAPE_MK3.py --device cuda # This one used xfeat


python3 VAPE_XFEAT.py
# 또는
python3 VAPE_XFEAT.py --target-fps 25 --performance-mode


python3 VAPE_MK1.py --device cuda # This one used superpoint



# 20250618

python3 annotator_config.py --reference-folder references/ --num-keypoints 6

python3 AutoMK3.py     --reference-config interactive_reference_config.json     --input-folder anno_images/     --visualize 


# 20250619

python3 VAPE_MK42.py



# 202506124

python3 annotator_config.py --reference-folder references/ --num-keypoints 6

python3 AutoMK2.py     --reference-config interactive_reference_config.json     --input-folder anno_images/     --visualize 


python3 AutoMK4.py     --reference-config interactive_reference_config.json     --input-folder anno_images/   --selection-method sam_segment_based



python3 AutoMK5.py     --reference-config interactive_reference_config.json     --input-folder anno_images/ --text-prompt "white aircraft"     --yolo-confidence 0.3     --sam-model sam2_b.pt     --debug


python3 annotaion_setup.py     --reference-folder references/     --num-keypoints 6     --text-prompt "white aircraft"


#20250625

python3 VAPE_MK43.py



#20250626

python3 VAPE_MK45_ONNX.py

python3 VAPE_MK45_debug_ONNX.py

python3 onnx_pose.py


#20250630

python3 annotator_config.py --reference-folder references/ --num-keypoints 14

python3 AutoMK3.py     --reference-config interactive_reference_config.json     --input-folder anno_images/     --visualize 

python3 AutoMK43.py     --reference-config interactive_reference_config.json     --input-folder anno_images/     --visualize --bbox-mode minimal



# Camera Mode (Original)
python3 VAPE_MK43_with_JSON_Export.py --save_pose results.json

# Batch Mode (Like main_sync.py)
python3 VAPE_MK43_JSONver.py --image_dir assets/Ruun_images/ICUAS/20250210/extracted_images_test5/ --csv_file assets/Ruun_images/ICUAS/20250210/extracted_images_test5/image_index.csv --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --save_pose results_20250702.json


# 20250705

python3 AutoMK7.py     --reference-config interactive_reference_config.json     --input-folder /media/runbk0401/Storage5/RUUN_GLUE_DATABASE/assets/Ruun_images/ICUAS/merged/     --ransac-threshold 10 --distance-threshold 10 --min-matches-for-ransac 8 --debug --crop-padding-ratio 0.16



#20250706
python3 VAPE_MK43_JSONver.py --image_dir /media/runbk0401/Storage5/RUUN_GLUE_DATABASE/REF-20250124/extracted_images_20250124_ICUAS1/ --csv_file /media/runbk0401/Storage5/RUUN_GLUE_DATABASE/REF-20250124/extracted_images_20250124_ICUAS1/image_index.csv --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --save_pose results_20250706_ICUAS.json


python3 VAPE_MK43_JSONver.py --image_dir /media/runbk0401/Storage5/RUUN_GLUE_DATABASE/assets/Ruun_images/ICUAS/20250128/extracted_images_20250128_test2/ --csv_file /media/runbk0401/Storage5/RUUN_GLUE_DATABASE/assets/Ruun_images/ICUAS/20250128/extracted_images_20250128_test2/image_index.csv --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --save_pose results_20250706_ICUAS.json


python3 main_sync.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --resize 1280 720 --image_dir /media/runbk0401/Storage5/RUUN_GLUE_DATABASE/assets/Ruun_images/ICUAS/20250128/extracted_images_20250128_test2/ --csv_file /media/runbk0401/Storage5/RUUN_GLUE_DATABASE/assets/Ruun_images/ICUAS/20250128/extracted_images_20250128_test2/image_index.csv --save_pose 20250706_20250128_test2.json

python3 VAPE_MK45_Robust.py --image_dir /media/runbk0401/Storage5/RUUN_GLUE_DATABASE/assets/Ruun_images/ICUAS/20250128/extracted_images_20250128_test2/ --csv_file /media/runbk0401/Storage5/RUUN_GLUE_DATABASE/assets/Ruun_images/ICUAS/20250128/extracted_images_20250128_test2/image_index.csv 


# 20250707

python3 VAPE_Journal.py --video_file simple_indoor.mp4

python3 VAPE_MK46.py --image_dir /media/runbk0401/Storage5/RUUN_GLUE_DATABASE/assets/Ruun_images/ICUAS/20250128/extracted_images_20250128_test2/ --csv_file /media/runbk0401/Storage5/RUUN_GLUE_DATABASE/assets/Ruun_images/ICUAS/20250128/extracted_images_20250128_test2/image_index.csv 

# 20250708

python3 VAPE_MK46.py 

python3 VAPE_MK47.py 

python3 AnchorTool.py --reference assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --new_anchor NE.png --reference_viewpoint NE

python3 AnchorTool.py --reference assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --new_anchor SE.png --reference_viewpoint SE



######### CAMERA INTRINSICS ###########

# # Calib_Phone_Opti
        # focal_length_x = 1078.06451
        # focal_length_y = 1081.77221
        # cx = 628.078538
        # cy = 362.156441

        # distCoeffs = np.array(
        #     [5.63748710e-02, -7.51721332e-01, -6.97952865e-04, -3.84299642e-03,6.18234012e+00],
        #     dtype=np.float32
        # )



	# # Camera calibration parameters - DEFAULT
        # focal_length_x = 1430.10150
        # focal_length_y = 1430.48915
        # cx = 640.85462
        # cy = 480.64800

        # distCoeffs = np.array([0.3393, 2.0351, 0.0295, -0.0029, -10.9093], dtype=np.float32)


        ## Calib_webcam ICUAS LAB 20250124
        focal_length_x = 1460.10150  # fx from the calibrated camera matrix
        focal_length_y = 1456.48915  # fy from the calibrated camera matrix
        cx = 604.85462               # cx from the calibrated camera matrix
        cy = 328.64800               # cy from the calibrated camera matrix

        distCoeffs = np.array(
            [3.56447550e-01, -1.09206851e+01, 1.40564820e-03, -1.10856449e-02, 1.20471120e+02],
            dtype=np.float32
        )

        distCoeffs = None


        # ## DJI calibration
        # ## Calib_camera 20250414

        # focal_length_x = 1313.36340#1.4065844e+03#1313.36340  # fx from the calibrated camera matrix
        # focal_length_y = 1313.36340#1.4065844e+03#1316.59196  # fy from the calibrated camera matrix
        # cx = 955.924592             # cx from the calibrated camera matrix
        # cy = 583.905916             # cy from the calibrated camera matrix

        # distCoeffs = np.array(
        #     [2.33571678e-01, -1.63149722e+00, 8.52446663e-03, 2.40400897e-04, 2.83400276e+00],
        #     dtype=np.float32
        # )

        # #distCoeffs = None



        # ## 982 514 guess

        # focal_length_x = 4049.6#2686.0
        # focal_length_y = 4037.6#2686.0
        # cx = 740.0#491.0              # cx from the calibrated camera matrix
        # cy = 415.0#257.0                 # cy from the calibrated camera matrix

        # # distCoeffs = np.array(
        # #     [2.33571678e-01, -1.63149722e+00, 8.52446663e-03, 2.40400897e-04, 2.83400276e+00],
        # #     dtype=np.float32
        # # )

        # distCoeffs = None

########################################



