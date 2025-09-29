# IndexTTS API 使用说明

IndexTTS 提供了 RESTful API 接口用于文本转语音服务。本文档详细说明了如何正确调用这些接口。

## 启动 API 服务

在使用 API 之前，需要先启动服务：

```bash
cd index-tts
source .venv/bin/activate
python api.py
```
或者
```bash
uv run api.py [--port PORT] [--host HOST] [--model_dir MODEL_DIR] [--fp16] [--deepspeed] [--cuda_kernel]
```

参数说明：
- `--port`: 服务端口，默认为 8000
- `--host`: 服务主机地址，默认为 0.0.0.0
- `--model_dir`: 模型文件目录，默认为 ./checkpoints
- `--fp16`: 使用 FP16 进行推理（如果可用）
- `--deepspeed`: 使用 DeepSpeed 加速（如果可用）
- `--cuda_kernel`: 使用 CUDA 内核进行推理（如果可用）

## API 接口说明

### 1. 创建 TTS 任务（POST 方式）

**接口地址**: `POST /api/v1/tts/tasks`

**请求头**: `Content-Type: application/json`

**请求参数**:

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| text | string | 是 | - | 要合成的文本 |
| prompt_audio | string | 否 | "tests/sample_prompt.wav" | 音色参考音频路径 |
| return_audio | boolean | 否 | false | 是否直接返回音频 |
| emo_control_method | integer | 否 | 0 | 情感控制方法 (0: 与说话人相同, 1: 来自参考音频, 2: 来自向量, 3: 来自文本) |
| emo_ref_path | string | 否 | null | 情感参考音频路径 |
| emo_weight | float | 否 | 0.65 | 情感权重 |
| emo_text | string | 否 | null | 情感描述文本 |
| emo_vec | array | 否 | null | 情感向量 |
| emo_random | boolean | 否 | false | 是否使用随机情感 |
| max_text_tokens_per_segment | integer | 否 | 120 | 每段最大文本 token 数 |
| do_sample | boolean | 否 | true | 是否采样 |
| top_p | float | 否 | 0.8 | 核采样参数 |
| top_k | integer | 否 | 30 | top-k 采样参数 |
| temperature | float | 否 | 0.8 | 温度参数 |
| length_penalty | float | 否 | 0.0 | 长度惩罚 |
| num_beams | integer | 否 | 3 | beam search 数量 |
| repetition_penalty | float | 否 | 10.0 | 重复惩罚 |
| max_mel_tokens | integer | 否 | 1500 | 最大 mel token 数 |

**请求示例**:

```bash
# 异步任务模式
curl -X POST "http://localhost:8000/api/v1/tts/tasks" \
     -H "Content-Type: application/json" \
     -d '{
           "text": "你好，欢迎使用 IndexTTS 语音合成系统",
           "prompt_audio": "path/to/speaker.wav"
         }'

# 直接返回音频模式
curl -X POST "http://localhost:8000/api/v1/tts/tasks" \
     -H "Content-Type: application/json" \
     -d '{
           "text": "你好，欢迎使用 IndexTTS 语音合成系统",
           "prompt_audio": "path/to/speaker.wav",
           "return_audio": true
         }' \
     --output result.wav
```

**响应示例**:

```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "Task created, waiting to be processed",
  "result_path": null
}
```

### 2. 创建 TTS 任务（GET 方式）

**接口地址**: `GET /api/v1/tts/tasks`

此接口提供简化的同步调用方式，适用于快速合成场景。

**请求参数**:

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| text | string | 是 | - | 要合成的文本 |
| prompt_audio | string | 否 | "tests/sample_prompt.wav" | 音色参考音频路径 |

**请求示例**:

```bash
curl "http://localhost:8000/api/v1/tts/tasks?text=你好，欢迎使用IndexTTS&prompt_audio=path/to/speaker.wav" \
     -o result.wav
```

### 3. 查询任务状态

**接口地址**: `GET /api/v1/tts/tasks/{task_id}`

**请求示例**:

```bash
curl "http://localhost:8000/api/v1/tts/tasks/550e8400-e29b-41d4-a716-446655440000"
```

**响应示例**:

```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "message": "Task completed successfully",
  "result_path": "outputs/tasks/550e8400-e29b-41d4-a716-446655440000.wav"
}
```

### 4. 获取任务结果

**接口地址**: `GET /api/v1/tts/tasks/{task_id}/result`

当任务状态为 completed 时，可以通过此接口获取合成的音频文件。

**请求示例**:

```bash
curl "http://localhost:8000/api/v1/tts/tasks/550e8400-e29b-41d4-a716-446655440000/result" \
     -o result.wav
```

## 重要注意事项

1. **请求格式**：所有 POST 请求必须使用 `application/json` 格式发送数据，不要使用 `multipart/form-data` 格式。

2. **文件路径**：确保 [prompt_audio](file:///TTS/index-tts/api.py#L68-L68) 和 [emo_ref_path](file:///TTS/index-tts/api.py#L71-L71) 指向有效的音频文件路径。

3. **异步处理**：默认情况下，任务以异步方式处理。如果需要立即获得结果，可以设置 [return_audio](file:///TTS/index-tts/api.py#L91-L91) 为 true。

4. **任务状态**：任务可能的状态包括：
   - `pending`：任务已创建，等待处理
   - `processing`：任务正在处理中
   - `completed`：任务已完成
   - `failed`：任务处理失败

5. **错误处理**：API 会返回标准 HTTP 状态码和详细的错误信息，便于调试和错误处理。