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

python3 main_track.py --anchor assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png --resize 1280 720 --image_dir assets/Ruun_images/ICUAS/20250128/extracted_images_20250128_test4/ --csv_file assets/Ruun_images/ICUAS/20250128/extracted_images_20250128_test4/image_index.csv --save_pose 20250304_20250128_test4_pixel.json
