#!/usr/bin/env python3
"""
XFeat Integration Test Script
Test the XFeat + LighterGlue integration before running the full pose estimation pipeline.
"""

import torch
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

def test_xfeat_installation():
    """Test if XFeat can be loaded and basic functionality works"""
    print("🔄 Testing XFeat installation...")
    
    try:
        # Load XFeat model
        xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained=True, top_k=1024)
        print("✅ XFeat model loaded successfully!")
        
        # Check if CUDA is available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        xfeat.to(device)
        print(f"✅ XFeat moved to device: {device}")
        
        return xfeat, device
        
    except Exception as e:
        print(f"❌ XFeat loading failed: {e}")
        print("💡 Try installing dependencies: pip install kornia kornia-rs --no-deps")
        return None, None

def test_basic_feature_extraction(xfeat, device):
    """Test basic feature extraction on a simple image"""
    print("\n🔄 Testing basic feature extraction...")
    
    try:
        # Create a test image
        test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Add some features (circles and lines)
        cv2.circle(test_img, (200, 200), 50, (255, 255, 255), 2)
        cv2.circle(test_img, (400, 300), 30, (255, 255, 255), 2)
        cv2.line(test_img, (100, 100), (500, 400), (255, 255, 255), 2)
        
        start_time = time.time()
        
        # Extract features
        with torch.no_grad():
            output = xfeat.detectAndCompute(test_img, top_k=1024)[0]
            output.update({'image_size': (test_img.shape[1], test_img.shape[0])})
        
        extraction_time = (time.time() - start_time) * 1000
        
        # Check output
        keypoints = output['keypoints']
        num_keypoints = len(keypoints)
        
        print(f"✅ Feature extraction successful!")
        print(f"   - Keypoints detected: {num_keypoints}")
        print(f"   - Extraction time: {extraction_time:.1f}ms")
        print(f"   - Keypoints shape: {keypoints.shape}")
        
        return test_img, output
        
    except Exception as e:
        print(f"❌ Feature extraction failed: {e}")
        return None, None

def test_matching(xfeat, device):
    """Test feature matching between two images"""
    print("\n🔄 Testing feature matching...")
    
    try:
        # Create two test images with some common features
        img1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Add common features
        for center in [(200, 200), (400, 300), (300, 150)]:
            cv2.circle(img1, center, 30, (255, 255, 255), 2)
            # Slightly shifted in second image
            cv2.circle(img2, (center[0] + 10, center[1] + 5), 30, (255, 255, 255), 2)
        
        start_time = time.time()
        
        # Extract features from both images
        with torch.no_grad():
            output1 = xfeat.detectAndCompute(img1, top_k=1024)[0]
            output1.update({'image_size': (img1.shape[1], img1.shape[0])})
            
            output2 = xfeat.detectAndCompute(img2, top_k=1024)[0]
            output2.update({'image_size': (img2.shape[1], img2.shape[0])})
            
            # Match features using LighterGlue
            mkpts0, mkpts1,_ = xfeat.match_lighterglue(output1, output2)
        
        matching_time = (time.time() - start_time) * 1000
        
        num_matches = len(mkpts0)
        
        print(f"✅ Feature matching successful!")
        print(f"   - Matches found: {num_matches}")
        print(f"   - Total time: {matching_time:.1f}ms")
        print(f"   - Match points shape: {mkpts0.shape}, {mkpts1.shape}")
        
        return img1, img2, mkpts0, mkpts1
        
    except Exception as e:
        print(f"❌ Feature matching failed: {e}")
        return None, None, None, None

def test_performance_benchmark(xfeat, device, num_iterations=10):
    """Benchmark XFeat performance"""
    print(f"\n🔄 Running performance benchmark ({num_iterations} iterations)...")
    
    extraction_times = []
    matching_times = []
    
    try:
        for i in range(num_iterations):
            # Create test images
            img1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            img2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Add some features
            cv2.circle(img1, (200, 200), 30, (255, 255, 255), 2)
            cv2.circle(img2, (210, 205), 30, (255, 255, 255), 2)
            
            # Time feature extraction
            start_time = time.time()
            with torch.no_grad():
                output1 = xfeat.detectAndCompute(img1, top_k=1024)[0]
                output1.update({'image_size': (img1.shape[1], img1.shape[0])})
                
                output2 = xfeat.detectAndCompute(img2, top_k=1024)[0]
                output2.update({'image_size': (img2.shape[1], img2.shape[0])})
            
            extraction_time = (time.time() - start_time) * 1000
            extraction_times.append(extraction_time)
            
            # Time matching
            start_time = time.time()
            with torch.no_grad():
                mkpts0, mkpts1 = xfeat.match_lighterglue(output1, output2)
            
            matching_time = (time.time() - start_time) * 1000
            matching_times.append(matching_time)
            
            if (i + 1) % 5 == 0:
                print(f"   Progress: {i + 1}/{num_iterations}")
        
        avg_extraction = np.mean(extraction_times)
        avg_matching = np.mean(matching_times)
        avg_total = avg_extraction + avg_matching
        
        print(f"✅ Performance benchmark completed!")
        print(f"   - Average extraction time: {avg_extraction:.1f}ms")
        print(f"   - Average matching time: {avg_matching:.1f}ms")
        print(f"   - Average total time: {avg_total:.1f}ms")
        print(f"   - Theoretical max FPS: {1000/avg_total:.1f}")
        
        return extraction_times, matching_times
        
    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")
        return None, None

def test_camera_integration(xfeat, device, camera_id=0):
    """Test XFeat with real camera input"""
    print(f"\n🔄 Testing camera integration (camera ID: {camera_id})...")
    print("Press 'q' to quit, 's' to save test image")
    
    try:
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"❌ Could not open camera {camera_id}")
            return False
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        frame_count = 0
        fps_times = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to capture frame")
                break
            
            frame_count += 1
            start_time = time.time()
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Extract features
            with torch.no_grad():
                output = xfeat.detectAndCompute(frame_rgb, top_k=512)[0]
                keypoints = output['keypoints'].detach().cpu().numpy()
            
            processing_time = (time.time() - start_time) * 1000
            fps_times.append(processing_time)
            
            # Visualize keypoints
            display_frame = frame.copy()
            for kp in keypoints:
                cv2.circle(display_frame, (int(kp[0]), int(kp[1])), 2, (0, 255, 0), -1)
            
            # Add performance info
            if len(fps_times) > 0:
                avg_time = np.mean(fps_times[-10:])  # Average of last 10 frames
                fps = 1000 / avg_time if avg_time > 0 else 0
                cv2.putText(display_frame, f"XFeat FPS: {fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Keypoints: {len(keypoints)}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Time: {processing_time:.1f}ms", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('XFeat Camera Test', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(f'xfeat_test_frame_{frame_count}.jpg', display_frame)
                print(f"Saved test frame: xfeat_test_frame_{frame_count}.jpg")
        
        cap.release()
        cv2.destroyAllWindows()
        
        if len(fps_times) > 10:
            avg_time = np.mean(fps_times[5:])  # Skip first few frames
            print(f"✅ Camera integration test completed!")
            print(f"   - Average processing time: {avg_time:.1f}ms")
            print(f"   - Average FPS: {1000/avg_time:.1f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Camera integration test failed: {e}")
        return False

def main():
    """Run all XFeat integration tests"""
    print("🚀 XFeat Integration Test Suite")
    print("=" * 50)
    
    # Test 1: Installation
    xfeat, device = test_xfeat_installation()
    if xfeat is None:
        print("❌ Cannot proceed with tests - XFeat installation failed")
        return
    
    # Test 2: Basic feature extraction
    test_img, output = test_basic_feature_extraction(xfeat, device)
    if output is None:
        print("❌ Cannot proceed with tests - Feature extraction failed")
        return
    
    # Test 3: Feature matching
    img1, img2, mkpts0, mkpts1 = test_matching(xfeat, device)
    if mkpts0 is None:
        print("❌ Feature matching test failed")
        return
    
    # Test 4: Performance benchmark
    extraction_times, matching_times = test_performance_benchmark(xfeat, device)
    
    # Test 5: Camera integration (optional)
    try_camera = input("\n🎥 Test with camera? (y/N): ").lower().strip() == 'y'
    if try_camera:
        camera_success = test_camera_integration(xfeat, device)
    
    print("\n✅ All tests completed!")
    print("🎯 XFeat integration is ready for pose estimation pipeline")

if __name__ == "__main__":
    main()