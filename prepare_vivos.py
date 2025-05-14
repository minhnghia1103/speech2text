import os
import csv
import soundfile as sf
from datasets import load_dataset
from speechbrain.dataio.dataio import read_audio_info
import traceback

def create_csv_from_dataset(dataset, save_folder, split_name, temp_wav_folder):
    """Tạo file CSV từ dataset VIVOS"""
    
    csv_file = os.path.join(save_folder, f"{split_name}.csv")
    header = ["ID", "duration", "wav", "spk_id", "wrd"]
    
    # Đảm bảo thư mục tồn tại
    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(temp_wav_folder, exist_ok=True)
    
    # Kiểm tra dataset
    if len(dataset[split_name]) == 0:
        print(f"ERROR: Dataset split '{split_name}' không chứa dữ liệu!")
        return None
    
    # In thông tin về số lượng mẫu
    print(f"Đang xử lý {len(dataset[split_name])} mẫu trong split '{split_name}'")
    
    # In mẫu đầu tiên để debug
    first_example = dataset[split_name][0]
    print(f"Mẫu đầu tiên: {first_example.keys()}")
    
    success_count = 0
    error_count = 0
    
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        for idx, example in enumerate(dataset[split_name]):
            try:
                snt_id = f"VIVOS_{split_name.upper()}_{idx+1:03d}"
                wav_path = os.path.join(temp_wav_folder, f"{snt_id}.wav")
                
                # Xử lý trường audio
                if "audio" not in example:
                    print(f"Bỏ qua mẫu {idx}: Không có trường audio")
                    error_count += 1
                    continue
                
                # Kiểm tra cấu trúc audio
                if isinstance(example["audio"], dict) and "array" in example["audio"] and "sampling_rate" in example["audio"]:
                    audio = example["audio"]["array"]
                    sample_rate = example["audio"]["sampling_rate"]
                else:
                    print(f"Cấu trúc audio không đúng định dạng ở mẫu {idx}")
                    print(f"Cấu trúc audio: {type(example['audio'])}")
                    if isinstance(example["audio"], dict):
                        print(f"Khóa trong audio: {example['audio'].keys()}")
                    error_count += 1
                    continue
                
                # Ghi file âm thanh
                sf.write(wav_path, audio, sample_rate)
                
                # Đọc thông tin duration
                try:
                    duration = read_audio_info(wav_path).duration
                except Exception as e:
                    print(f"Lỗi khi đọc thông tin audio: {e}")
                    duration = len(audio) / sample_rate  # Ước tính duration
                
                # Xử lý raw_raw_transcription
                if "raw_transcription" not in example:
                    # print(f"Không tìm thấy trường raw_transcription ở mẫu {idx}")
                    if "raw_transcription" in example:
                        wrd = example["raw_transcription"].lower()
                        # print(f"Sử dụng trường raw_transcription thay thế")
                    else:
                        # print(f"Không tìm thấy raw_transcription, bỏ qua mẫu {idx}")
                        error_count += 1
                        continue
                else:
                    wrd = example["raw_transcription"].lower()
                
                # Speaker ID
                spk_id = f"VIVOS_{split_name.upper()}"
                
                # Ghi vào CSV
                writer.writerow([snt_id, duration, wav_path, spk_id, wrd])
                success_count += 1
                
                # # In thông tin tiến độ
                # if (idx + 1) % 10 == 0:
                #     print(f"Đã xử lý {idx + 1} mẫu, thành công: {success_count}, lỗi: {error_count}")
            
            except Exception as e:
                print(f"Lỗi khi xử lý mẫu {idx}:")
                print(traceback.format_exc())
                error_count += 1
                continue
    
    # print(f"Hoàn thành xử lý split '{split_name}':")
    # print(f"Tổng số mẫu: {len(dataset[split_name])}")
    # print(f"Thành công: {success_count}")
    # print(f"Lỗi: {error_count}")
    
    if success_count == 0:
        print("CẢNH BÁO: Không có mẫu nào được xử lý thành công!")
        return None
    
    return csv_file

def extract_text_for_bpe(csv_path, output_text_file):
    """Trích xuất văn bản từ CSV để huấn luyện mô hình BPE"""
    if not csv_path or not os.path.exists(csv_path):
        print(f"ERROR: File CSV không tồn tại: {csv_path}")
        return
    
    text_count = 0
    with open(output_text_file, "w", encoding="utf-8") as f:
        with open(csv_path, "r", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                if "wrd" in row and row["wrd"]:
                    f.write(row["wrd"] + "\n")
                    text_count += 1
    
    print(f"Đã trích xuất {text_count} dòng văn bản vào file {output_text_file}")

def prepare_vivos_hf(dataset, save_folder, temp_wav_folder="temp_wavs", dev_size=0.1, splits=["train", "dev", "test"]):
    """Chuẩn bị dataset VIVOS từ Hugging Face cho SpeechBrain"""
    print(f"Bắt đầu chuẩn bị dataset VIVOS")
    print(f"Splits có sẵn: {list(dataset.keys())}")
    
    # Tạo thư mục save_folder nếu chưa tồn tại
    os.makedirs(save_folder, exist_ok=True)
    
    # Tạo tập validation nếu cần
    if "dev" in splits and "dev" not in dataset:
        print(f"Tạo split 'dev' từ 'train' với tỷ lệ {dev_size}")
        split_dict = dataset["train"].train_test_split(test_size=dev_size, seed=42)
        dataset = dataset.copy()
        dataset["train"] = split_dict["train"]
        dataset["dev"] = split_dict["test"]
        print(f"Kích thước train mới: {len(dataset['train'])}, dev: {len(dataset['dev'])}")
    
    # Xử lý từng split
    for split in splits:
        if split in dataset:
            print(f"\n--- Xử lý split: {split} ---")
            csv_file = create_csv_from_dataset(dataset, save_folder, split, os.path.join(temp_wav_folder, split))
            
            if csv_file:
                output_text_file = os.path.join(save_folder, f"{split}_bpe.txt")
                extract_text_for_bpe(csv_file, output_text_file)
                
                # Kiểm tra file đầu ra
                if os.path.exists(output_text_file):
                    with open(output_text_file, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                    print(f"File {output_text_file} có {len(lines)} dòng")
                else:
                    print(f"CẢNH BÁO: File {output_text_file} không được tạo!")
            else:
                print(f"Bỏ qua tạo file BPE cho split '{split}' do không có CSV")
        else:
            print(f"Split '{split}' không có trong dataset, bỏ qua")

if __name__ == "__main__":
    print("Đang tải dataset VIVOS từ Hugging Face...")
    try:
        dataset = load_dataset("ademax/vivos-vie-speech2text")
        print(f"Dataset loaded successfully with keys: {list(dataset.keys())}")
        
        # Kiểm tra cấu trúc dataset
        for split in dataset:
            if len(dataset[split]) > 0:
                print(f"Split '{split}' có {len(dataset[split])} mẫu")
                first_item = dataset[split][0]
                print(f"Cấu trúc mẫu đầu tiên trong '{split}': {list(first_item.keys())}")
        
        # Gọi hàm với dev_size để lấy phần từ tập train
        prepare_vivos_hf(dataset, save_folder="results/vivos", temp_wav_folder="temp_wavs", dev_size=0.1, splits=["train", "dev"])
        
        # Kiểm tra kết quả cuối cùng
        result_dir = "results/vivos"
        if os.path.exists(result_dir):
            files = os.listdir(result_dir)
            print(f"\nCác file được tạo trong thư mục {result_dir}:")
            for file in files:
                file_path = os.path.join(result_dir, file)
                file_size = os.path.getsize(file_path)
                print(f"- {file}: {file_size} bytes")
        else:
            print(f"Thư mục kết quả {result_dir} không tồn tại!")
            
    except Exception as e:
        print(f"Lỗi khi tải hoặc xử lý dataset:")
        print(traceback.format_exc())