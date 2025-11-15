import cv2
import os

from core.algorithm import (
    sift_align_images,
    crop_pcb_from_color,
    paste_crop_to_canvas,
    yolo_infer,
    check_shift_orb
)

from camera import BaslerCamera


def main_pipeline():

    # --- 1. Connect camera ---
    cam1 = BaslerCamera(save_folder="avi_project/stored_images/raw_images")
    cam1.connect()
    image_path = cam1.capture_and_save(".bmp")
    cam1.disconnect()

    # --- 2. Align ---
    golden_path = "avi_project/golden/golden_ok.bmp"
    golden_img = cv2.imread(golden_path)
    raw_img = cv2.imread(image_path)

    if golden_img is None:
        print("Không đọc được golden image:", golden_path)
        return

    print("Align ảnh...")
    aligned_img, H = sift_align_images(golden_img, raw_img, draw_matches=False)

    # --- 3. Crop PCB ---
    print("Crop PCB...")
    cropped_img = crop_pcb_from_color(aligned_img, debug=False)

    # --- 4. Canvas ---
    print("Paste lên canvas...")
    final_img = paste_crop_to_canvas(cropped_img, debug=False)

    fname = os.path.basename(image_path)
    final_path = os.path.join("avi_project/stored_images/final_images", fname)
    cv2.imwrite(final_path, final_img)

    print("Final image saved:", final_path)

    # --- 5. YOLO detect ---
    print("YOLO infer...")
    results, out_img = yolo_infer(
        model_path="runs/detect/train/weights/best.pt",
        img_path=final_path,
        save_folder="avi_project/stored_images/yolo_images",
        debug=False
    )

    print("YOLO boxes:", results[0].boxes.data)
    print("YOLO output saved:", out_img)

    # --- 6. Check shift ---
    print("Kiểm tra lệch ORB...")
    golden_img = cv2.imread(golden_path)
    test_img = cv2.imread(final_path)

    overlay = test_img.copy()

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        bbox = (x1, y1, x2, y2)

        shift_result = check_shift_orb(golden_img, test_img, bbox)
        print("Shift:", shift_result)



if __name__ == "__main__":
    main_pipeline()
