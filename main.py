from aoi import capture_one_basler, yolo_color, yolo_infer, connect_basler, grab_one_frame, sift_align_images, crop_pcb_from_color, paste_crop_to_canvas, check_shift_orb
from aoi import draw_orb_overlay, check_shift_sift
import cv2
import os


if __name__ == "__main__":

    cam, converter = connect_basler()

    # 1. Take a picture
    ok, raw_path = grab_one_frame(cam, converter)  # lấy luôn numpy img

    if not ok:
        print("lỗi chụp ảnh")
        exit()

    print("Đã chụp:", raw_path)

    cam.Close()  # nhớ đóng

    # 2. Align image
    golden_img = cv2.imread("golden/golden_ok.bmp")
    raw_img = cv2.imread(raw_path)
    if golden_img is None:
        print("Không đọc được golden ảnh")
        exit()

    aligned_img, H = sift_align_images(golden_img, raw_img, draw_matches=False)

    # 3. Crop image
    cropped_img = crop_pcb_from_color(aligned_img, debug=False)

    # 4. Paste to canvas
    final_img = paste_crop_to_canvas(cropped_img, debug=False)

    os.makedirs("final_images", exist_ok=True)
    fname = os.path.basename(raw_path)
    final_path = os.path.join("final_images", fname)
    cv2.imwrite(final_path, final_img)
    print("Final image saved:", final_path)

    # 5. Detect with yolo
    results, out_img = yolo_infer(
        model_path="runs/detect/train/weights/best.pt",
        img_path= final_path,  # YOLO cần path vẫn OK
        debug = False
    )

    print("Kết quả:", results[0].boxes.data)
    print("Ảnh lưu:", out_img)

    # 6. Feature detect/so sánh lệch trong box
    golden_img = cv2.imread("golden/golden_ok.bmp")
    test_img = cv2.imread(final_path)

    overlay = test_img.copy()  # ✅ tạo ảnh overlay trước

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        bbox = (x1, y1, x2, y2)  # ✅ tạo bbox đúng

        r = check_shift_orb(golden_img, test_img, bbox)
        print(r)

        overlay = draw_orb_overlay(overlay, bbox, r)  # ✅ dùng bbox

    cv2.imwrite("orb_result.bmp", overlay)
    print("Saved orb_result.bmp")


