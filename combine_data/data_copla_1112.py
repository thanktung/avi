import pandas as pd
import re, glob, os

def parse_file(path, tray_no):
    """Đọc 1 file đo và trả về DataFrame Part + 9 Ball"""
    records = []
    current_part = None
    current_balls = {}

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            # Nhận Part #: x
            m_part = re.search(r'Part\s*#:\s*(\d+)', line)
            if m_part:
                # Nếu có part trước đó -> lưu lại
                if current_part is not None:
                    row = {'Part': current_part, 'Tray': tray_no}
                    row.update({f'Ball{i}': current_balls.get(i, 0.0) for i in range(1,10)})
                    records.append(row)
                current_part = int(m_part.group(1))
                current_balls = {}
                continue

            # Nhận dòng Ball
            m_ball = re.match(r'^\s*(\d+)\s+([\d.]+)', line)
            if m_ball and current_part is not None:
                b = int(m_ball.group(1))
                wrp = float(m_ball.group(2))
                current_balls[b] = wrp

        # Lưu part cuối cùng
        if current_part is not None:
            row = {'Part': current_part, 'Tray': tray_no}
            row.update({f'Ball{i}': current_balls.get(i, 0.0) for i in range(1,10)})
            records.append(row)

    return pd.DataFrame(records)

if __name__ == "__main__":
    # 📁 đọc tất cả file trong thư mục (sửa lại cho đúng)
    files = sorted(glob.glob("data/*.txt"))
    if not files:
        print("⚠️ Không tìm thấy file trong combine_data/data/")
        exit()

    dfs = []
    for i, f in enumerate(files, 1):
        df = parse_file(f, i)
        print(f"{os.path.basename(f)}: {len(df)} parts")
        dfs.append(df)

    # Gộp toàn bộ Tray
    df_all = pd.concat(dfs, ignore_index=True)

    # Xuất CSV
    cols = ['Part', 'Tray'] + [f'Ball{i}' for i in range(1,10)]
    df_all = df_all[cols].sort_values(['Part','Tray'])
    df_all.to_csv("summary_measurements.csv", index=False, float_format="%.2f", encoding="utf-8-sig")
    print("✅ Saved summary_measurements.csv")
