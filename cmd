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