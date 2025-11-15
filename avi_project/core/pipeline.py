import cv2
import os

from aoi import (
    connect_basler,
    grab_one_frame,
    sift_align_images,
    crop_pcb_from_color,
    paste_crop_to_canvas,
    yolo_infer,
    check_shift_orb,
    draw_orb_overlay
)


def main():
    # --- 1. Connect camera ---
    cam, converter = connect_basler()
    ok, raw_path = grab_one_frame(cam, converter)

    if not ok:
        print("❌ Lỗi chụp ảnh")
        return

    print("📷 Đã chụp:", raw_path)
    cam.Close()

    # --- 2. Align ---
    golden_path = "golden/golden_ok.bmp"
    golden_img = cv2.imread(golden_path)
    raw_img = cv2.imread(raw_path)

    if golden_img is None:
        print("❌ Không đọc được golden image:", golden_path)
        return

    print("🔧 Align ảnh...")
    aligned, H = sift_align_images(golden_img, raw_img, draw_matches=False)

    # --- 3. Crop PCB ---
    print("✂ Crop PCB...")
    cropped = crop_pcb_from_color(aligned, debug=False)

    # --- 4. Canvas ---
    print("🧩 Paste lên canvas...")
    final_img = paste_crop_to_canvas(cropped, debug=False)

    os.makedirs("final_images", exist_ok=True)
    fname = os.path.basename(raw_path)
    final_path = os.path.join("final_images", fname)
    cv2.imwrite(final_path, final_img)

    print("💾 Final image saved:", final_path)

    # --- 5. YOLO detect ---
    print("🤖 YOLO infer...")
    results, out_img = yolo_infer(
        model_path="runs/detect/train/weights/best.pt",
        img_path=final_path,
        debug=False
    )

    print("📦 YOLO boxes:", results[0].boxes.data)
    print("💾 YOLO output saved:", out_img)

    # --- 6. Check shift ---
    print("📐 Kiểm tra lệch ORB...")
    golden_img = cv2.imread(golden_path)
    test_img = cv2.imread(final_path)

    overlay = test_img.copy()

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        bbox = (x1, y1, x2, y2)

        shift_result = check_shift_orb(golden_img, test_img, bbox)
        print("Shift:", shift_result)

        overlay = draw_orb_overlay(overlay, bbox, shift_result)

    cv2.imwrite("orb_result.bmp", overlay)
    print("💾 Saved orb_result.bmp")


if __name__ == "__main__":
    main()
