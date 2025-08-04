# credits:
#   - ChatGPT (gpu multi process)
#   - https://github.com/codebox/mosaic (exact match)

import os
import numpy as np
from PIL import Image
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ---------- 參數設定 ----------
tile_dir = './bad_frames'
fit_dir = './bad_frames'
output_dir = './bad_fit'
os.makedirs(output_dir, exist_ok=True)

W, H = 480, 360 # bad apple: 480 x 360; sao3: 1280 x 720
W_SIZE, H_SIZE = 40, 40
thumb_w, thumb_h = W // W_SIZE, H // H_SIZE
plot_multiplier = 1.5
plot_w, plot_h = int(thumb_w * plot_multiplier), int(thumb_h * plot_multiplier)

print("sub full res:", W_SIZE * thumb_w, H_SIZE * thumb_h)

# ---------- test global -----------

class TileBuffer:
    def __init__(self):
        self.buffer = {}

    def set(self, serial, img):
        self.buffer[serial] = img

    def get(self, serial):
        return self.buffer.get(serial, None)

    def clear(self):
        self.buffer.clear()

# ---------- 載入圖像 ----------
def load_image_tensor(path, size):
    img = Image.open(path).convert('RGB')
    return torch.tensor(np.array(img.resize(size, Image.Resampling.LANCZOS)), dtype=torch.float32), img

def setup_tile_tensor_and_buffer(tile_buffer):
    print("Loading tile...")
    tile_serials = []
    tile_tensors = []

    # for sao3
    # for file in tqdm(os.listdir(tile_dir)):
    #     serial = file.rsplit('.', 1)[0]  # remove file extension
    #     img_path = f"{tile_dir}/{file}"
    #     img_tensor, img = load_image_tensor(img_path, (thumb_w, thumb_h))
    #     tile_serials.append(serial)
    #     tile_tensors.append(img_tensor)

    #     img = img.resize((plot_w, plot_h), Image.Resampling.LANCZOS)
    #     tile_buffer.set(serial, img)
    
    # for bad apple
    for i in tqdm(range(40, 6514 + 1, 1)):
        serial = str(i)
        img_path = f"{tile_dir}/{serial}.jpg"
        img_tensor = load_image_tensor(img_path, (thumb_w, thumb_h))
        tile_serials.append(serial)
        tile_tensors.append(img_tensor)

    tile_tensor = torch.stack(tile_tensors)
    return tile_serials, tile_tensor


# ---------- 切圖 ----------
def subdivide_target_tensor(image_tensor):
    subs = []
    for ih in range(H_SIZE):
        for iw in range(W_SIZE):
            y1 = ih * thumb_h
            x1 = iw * thumb_w
            y2 = y1 + thumb_h
            x2 = x1 + thumb_w
            crop = image_tensor[y1:y2, x1:x2, :]
            subs.append(crop)
    return torch.stack(subs)

# ---------- 匹配 ----------
def match_tiles_gpu(subs_tensor, tile_tensor, device, batch_size=200):
    tile_tensor = tile_tensor.to(device)
    tile_flat = tile_tensor.view(tile_tensor.size(0), -1, 3)
    all_best_indices = []

    for i in range(0, subs_tensor.size(0), batch_size):
        batch = subs_tensor[i:i+batch_size].to(device)
        batch_flat = batch.view(batch.size(0), -1, 3)

        with torch.no_grad():
            dists = ((batch_flat.unsqueeze(1) - tile_flat.unsqueeze(0)) ** 2).sum(dim=2).sum(dim=2)
            best_indices = dists.argmin(dim=1).cpu()
            all_best_indices.append(best_indices)

        torch.cuda.empty_cache()

    return torch.cat(all_best_indices, dim=0)

# ---------- 重建馬賽克圖 ----------
def reconstruct_image(best_indices, tile_serials, tile_buffer):
    out_img = Image.new('RGB', (W_SIZE * plot_w, H_SIZE * plot_h))
    for i, idx in enumerate(best_indices):
        tile_serial = tile_serials[idx]
        tile_img = tile_buffer.get(tile_serial)
        if tile_img is None:
            print(f"Warning: Tile {tile_serial} not found in buffer.")
            continue

        x = (i % W_SIZE) * plot_w
        y = (i // W_SIZE) * plot_h
        out_img.paste(tile_img, (x, y))
    return out_img

# ---------- 單 GPU 處理流程 ----------
def process_target(serial, tile_tensor, tile_serials, device, tile_buffer):
    img_path = os.path.join(fit_dir, f"{serial}.jpg")
    if not os.path.exists(img_path):
        return

    img = Image.open(img_path).convert('RGB').resize((W, H), Image.LANCZOS)
    img_tensor = torch.tensor(np.array(img), dtype=torch.float32).to(device)
    subs_tensor = subdivide_target_tensor(img_tensor)

    with torch.no_grad():
        best_indices = match_tiles_gpu(subs_tensor, tile_tensor, device=device)

    result = reconstruct_image(best_indices, tile_serials, tile_buffer)
    result.save(os.path.join(output_dir, f"{serial}.jpg"), "JPEG")

# ---------- 多 GPU worker ----------
def process_on_gpu(gpu_id, serials, tile_tensor_cpu, tile_serials, tile_buffer):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f'cuda:{gpu_id}')
    tile_tensor = tile_tensor_cpu.to(device)

    for serial in tqdm(serials, desc=f"[GPU {gpu_id}]"):
        try:
            process_target(serial, tile_tensor, tile_serials, device, tile_buffer)
        except Exception as e:
            print(f"[GPU {gpu_id}] Error processing {serial}: {e}")

# ---------- 分配任務並啟動 ----------
def run_multi_gpu(start_serial=0, end_serial=6571): # bad apple max: 6571, sao1 max: 2674
    num_gpus = torch.cuda.device_count()
    if num_gpus < 1:
        raise RuntimeError("No GPU available!")

    print(f"✅ Detected {num_gpus} GPUs, distributing workload...")

    tile_buffer = TileBuffer()
    tile_serials, tile_tensor_cpu = setup_tile_tensor_and_buffer(tile_buffer)

    all_serials = [str(i) for i in range(start_serial, end_serial + 1)]
    split_serials = [all_serials[i::num_gpus] for i in range(num_gpus)]

    ctx = mp.get_context('spawn')
    procs = []

    for i in range(num_gpus):
        p = ctx.Process(target=process_on_gpu, args=(i, split_serials[i], tile_tensor_cpu, tile_serials, tile_buffer))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

# ---------- 主程式 ----------
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    run_multi_gpu()
