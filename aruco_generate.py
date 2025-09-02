#!/usr/bin/env python3
"""
Generate A1-sized ChArUco board for ground truth calibration
A1 size: 594mm × 841mm (23.4" × 33.1")
"""

import cv2
import numpy as np

def generate_a1_charuco_board():
    """Generate ChArUco board optimized for A1 paper size"""
    
    # A1 dimensions in mm
    A1_WIDTH_MM = 594
    A1_HEIGHT_MM = 841
    
    # Calculate optimal grid size for A1
    # Target square size around 70-80mm for good distance detection
    target_square_mm = 75
    
    cols = int(A1_WIDTH_MM // target_square_mm)   # 7 columns
    rows = int(A1_HEIGHT_MM // target_square_mm)  # 11 rows
    
    # Adjust to fit A1 exactly
    square_size_mm = min(A1_WIDTH_MM / cols, A1_HEIGHT_MM / rows)
    marker_size_mm = square_size_mm * 0.75  # 75% of square size
    
    print(f"Generating ChArUco board:")
    print(f"  Grid: {cols}x{rows} ({cols*rows//2} markers)")
    print(f"  Square size: {square_size_mm:.1f}mm")
    print(f"  Marker size: {marker_size_mm:.1f}mm") 
    print(f"  Board size: {cols*square_size_mm:.1f}mm × {rows*square_size_mm:.1f}mm")
    
    # Create ArUco dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    
    # Create ChArUco board
    if hasattr(cv2.aruco, "CharucoBoard_create"):
        # Older OpenCV API
        board = cv2.aruco.CharucoBoard_create(
            cols, rows, 
            square_size_mm/1000.0,  # Convert to meters
            marker_size_mm/1000.0,  # Convert to meters  
            aruco_dict
        )
    else:
        # Newer OpenCV API
        board = cv2.aruco.CharucoBoard(
            (cols, rows),
            square_size_mm/1000.0,  # Convert to meters
            marker_size_mm/1000.0,  # Convert to meters
            aruco_dict
        )
    
    # Generate board image at high DPI for A1 printing
    # A1 at 300 DPI = 7016 × 9933 pixels
    # A1 at 600 DPI = 14032 × 19866 pixels (recommended for precision)
    dpi = 600
    img_width = int(A1_WIDTH_MM * dpi / 25.4)   # Convert mm to pixels at DPI
    img_height = int(A1_HEIGHT_MM * dpi / 25.4)
    
    print(f"  Image size: {img_width}x{img_height} pixels ({dpi} DPI)")
    
    # Draw the board - use the correct OpenCV function
    try:
        # Try newer API first
        board_img = cv2.aruco.CharucoBoard.generateImage(board, (img_width, img_height))
    except AttributeError:
        try:
            # Try alternative newer API
            board_img = board.generateImage((img_width, img_height))
        except AttributeError:
            # Fall back to older API
            board_img = cv2.aruco.drawCharucoBoard(board, (img_width, img_height))
    
    # Save the board
    filename = f"charuco_board_A1_{cols}x{rows}_{int(square_size_mm)}mm.png"
    cv2.imwrite(filename, board_img)
    print(f"  Saved: {filename}")
    
    # Also save a lower resolution version for quick viewing
    preview_img = cv2.resize(board_img, (1684, 2384))  # A1 at 72 DPI
    preview_filename = f"charuco_board_A1_{cols}x{rows}_{int(square_size_mm)}mm_preview.png"
    cv2.imwrite(preview_filename, preview_img)
    print(f"  Preview: {preview_filename}")
    
    # Generate calibration parameters for your code
    print(f"\nUse these parameters in your code:")
    print(f"cols={cols}, rows={rows},")
    print(f"square_len_m={square_size_mm/1000.0:.3f},")
    print(f"marker_len_m={marker_size_mm/1000.0:.3f},")
    print(f"dict_id=cv2.aruco.DICT_6X6_250")
    
    return board, cols, rows, square_size_mm/1000.0, marker_size_mm/1000.0

def generate_custom_size_board(width_mm, height_mm, target_square_mm=75):
    """Generate ChArUco board for custom paper size"""
    
    cols = int(width_mm // target_square_mm)
    rows = int(height_mm // target_square_mm)
    
    square_size_mm = min(width_mm / cols, height_mm / rows)
    marker_size_mm = square_size_mm * 0.75
    
    print(f"Custom board ({width_mm}×{height_mm}mm):")
    print(f"  Grid: {cols}x{rows}")
    print(f"  Square: {square_size_mm:.1f}mm")
    print(f"  Marker: {marker_size_mm:.1f}mm")
    
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    
    if hasattr(cv2.aruco, "CharucoBoard_create"):
        board = cv2.aruco.CharucoBoard_create(
            cols, rows,
            square_size_mm/1000.0,
            marker_size_mm/1000.0,
            aruco_dict
        )
    else:
        board = cv2.aruco.CharucoBoard(
            (cols, rows),
            square_size_mm/1000.0,
            marker_size_mm/1000.0,
            aruco_dict
        )
    
    # High resolution for printing
    dpi = 600
    img_width = int(width_mm * dpi / 25.4)
    img_height = int(height_mm * dpi / 25.4)
    
    board_img = board.draw((img_width, img_height))
    filename = f"charuco_board_{width_mm}x{height_mm}mm_{cols}x{rows}.png"
    cv2.imwrite(filename, board_img)
    print(f"  Saved: {filename}")
    
    return board

if __name__ == "__main__":
    print("ChArUco Board Generator for A1 Size")
    print("="*40)
    
    # Generate A1 board
    generate_a1_charuco_board()
    
    # Alternative: Generate for other common sizes
    print(f"\nOther common sizes:")
    print(f"A0 (841×1189mm): use generate_custom_size_board(841, 1189)")
    print(f"A2 (420×594mm): use generate_custom_size_board(420, 594)")
    print(f"A3 (297×420mm): use generate_custom_size_board(297, 420)")