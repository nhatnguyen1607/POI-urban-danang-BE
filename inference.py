import sys
import os
import json
import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, default='v4')
parser.add_argument('--text', type=str, required=True)
parser.add_argument('--image', type=str, default='')
args = parser.parse_args()

try:
    print(f"[DEBUG] Loading dependencies...", file=sys.stderr)
    
    import torch
    print(f"[DEBUG] PyTorch loaded: {torch.__version__}", file=sys.stderr)
    
    import torch.nn.functional as F
    from torchvision import transforms
    from PIL import Image
    import pandas as pd
    
    print(f"[DEBUG] All base dependencies loaded", file=sys.stderr)

    # Import encoder from local directory (no need to look in ../poi-urban-danang)
    from encoder.multimodal_encoder import MultimodalEncoder
    
    print(f"[DEBUG] MultimodalEncoder imported successfully", file=sys.stderr)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DEBUG] Device: {device}", file=sys.stderr)
    
    # Locate model path based on version
    model_dir = os.path.join(os.path.dirname(__file__), 'model', args.version, 'models_saved')
    print(f"[DEBUG] Looking for model in: {model_dir}", file=sys.stderr)
    
    # find the .pth file in this directory
    model_path = None
    if os.path.exists(model_dir):
        files = os.listdir(model_dir)
        print(f"[DEBUG] Files in model_dir: {files}", file=sys.stderr)
        for f in files:
            if f.endswith('.pth'):
                model_path = os.path.join(model_dir, f)
                print(f"[DEBUG] Found model file: {model_path}", file=sys.stderr)
                break
    else:
        print(f"[DEBUG] Model directory does not exist: {model_dir}", file=sys.stderr)
    
    if not model_path:
        raise Exception(f"No .pth model found in {model_dir}")

    # Load data for features
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'poi_data_ggmap.csv')
    print(f"[DEBUG] Loading data from: {data_path}", file=sys.stderr)
    df = pd.read_csv(data_path, encoding='utf-8')
    all_texts = df['LLM_Input_Text'].fillna('').tolist()
    print(f"[DEBUG] Loaded {len(all_texts)} POI texts", file=sys.stderr)

    print(f"[DEBUG] Creating MultimodalEncoder model...", file=sys.stderr)
    # Extract version number from args (v1, v2, v3, v4 -> 1, 2, 3, 4)
    version_num = int(args.version.replace('v', '')) if 'v' in args.version else int(args.version)
    model = MultimodalEncoder(version=version_num).to(device)
    print(f"[DEBUG] Model created successfully (version={version_num})", file=sys.stderr)
    
    print(f"[DEBUG] Loading model weights from {model_path}...", file=sys.stderr)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"[DEBUG] Model weights loaded successfully", file=sys.stderr)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if args.image and os.path.exists(args.image):
        img_pil = Image.open(args.image).convert("RGB")
        concept_img_tensor = transform(img_pil).unsqueeze(0).to(device)
    else:
        concept_img_tensor = torch.zeros((1, 3, 224, 224), dtype=torch.float32).to(device)

    with torch.no_grad():
        concept_feature = model(images=concept_img_tensor, texts=[args.text])
        
        # To avoid OOM and keep it fast, we process first 200 items for this demo
        # (In production we would precompute these features)
        all_poi_features = []
        batch_size = 32
        limit_texts = all_texts[:200] 
        
        for i in range(0, len(limit_texts), batch_size):
            batch_texts = limit_texts[i:i+batch_size]
            dummy_imgs = torch.zeros((len(batch_texts), 3, 224, 224), dtype=torch.float32).to(device)
            feats = model(images=dummy_imgs, texts=batch_texts)
            all_poi_features.append(feats)
            
        all_poi_features = torch.cat(all_poi_features, dim=0)
        similarities = F.cosine_similarity(concept_feature, all_poi_features, dim=-1)
        
        top_k = min(5, len(limit_texts))
        top_scores, top_indices = torch.topk(similarities, top_k)
        
        results = []
        for rank, (idx, score) in enumerate(zip(top_indices.cpu().numpy(), top_scores.cpu().numpy())):
            poi = df.iloc[idx]
            
            # Support both old column names (name, address, lat, lng) and new ones (Restaurant Name, District, Lat, Lon)
            poi_name = poi.get('name') or poi.get('Restaurant Name') or 'Unknown'
            poi_district = poi.get('District') or 'Đà Nẵng'
            if pd.isna(poi_name): poi_name = 'Unknown'
            if pd.isna(poi_district): poi_district = 'Đà Nẵng'
            
            # Address: try 'address' first, then fall back to District
            address = poi.get('address')
            if address and pd.notna(address):
                parts = str(address).split(',')
                district = parts[-3].strip() if len(parts) >= 3 else parts[-1].strip()
            else:
                district = str(poi_district)
            
            # Coordinates: try lowercase first, then capitalized
            lat_val = poi.get('lat') or poi.get('Lat') or 16.0544
            lon_val = poi.get('lng') or poi.get('Lon') or 108.2022
            if pd.isna(lat_val): lat_val = 16.0544
            if pd.isna(lon_val): lon_val = 108.2022
            
            # ID: try place_id first, then RestaurantID
            poi_id = poi.get('place_id') or poi.get('RestaurantID') or idx
            
            results.append({
                "id": str(poi_id),
                "name": f"{poi_name}",
                "district": district,
                "lat": float(lat_val),
                "lon": float(lon_val),
                "score": float(score * 100),
                "desc": str(poi.get('LLM_Input_Text', ''))[:150] + "..."
            })
            
        print(json.dumps(results))

except Exception as e:
    import traceback
    error_msg = str(e)
    tb_str = traceback.format_exc()
    print(f"[ERROR] {error_msg}", file=sys.stderr)
    print(f"[TRACEBACK]\n{tb_str}", file=sys.stderr)
    
    # Fallback to mock data if torch fails or model doesn't exist so frontend still works
    mock_results = [
        { "id": 1, "name": f"Mock: Khu vực Hải Châu (Version {args.version})", "score": 98.5, "district": "Hải Châu", "lat": 16.071, "lon": 108.245, "desc": f"Lỗi model: {error_msg[:150]}" },
        { "id": 2, "name": "Mock: Khu vực Sơn Trà", "score": 92.1, "district": "Sơn Trà", "lat": 16.035, "lon": 108.225, "desc": "Vui lòng kiểm tra log backend..." }
    ]
    print(json.dumps(mock_results))
