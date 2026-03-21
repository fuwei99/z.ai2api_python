# -*- coding: utf-8 -*-
"""
Z.ai 2 API
将 Z.ai 代理为 OpenAI Compatible 格式，支持免令牌、智能处理思考链、图片上传（仅登录后）等功能
基于 https://github.com/kbykb/OpenAI-Compatible-API-Proxy-for-Z 使用 AI 辅助重构。
"""

import os, json, re, requests, logging, uuid, base64
from datetime import datetime
from flask import Flask, request, Response, jsonify, make_response

from dotenv import load_dotenv
load_dotenv()

# 配置
BASE = str(os.getenv("BASE", "https://chat.z.ai"))
PORT = int(os.getenv("PORT", "8080"))
MODEL = str(os.getenv("MODEL", "GLM-4.5"))
TOKEN = str(os.getenv("TOKEN", "")).strip()
DEBUG_MODE = str(os.getenv("DEBUG", "false")).lower() == "true"
THINK_TAGS_MODE = str(os.getenv("THINK_TAGS_MODE", "reasoning"))
ANONYMOUS_MODE = str(os.getenv("ANONYMOUS_MODE", "true")).lower() == "true"

# tiktoken 预加载
cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tiktoken') + os.sep
os.environ["TIKTOKEN_CACHE_DIR"] = cache_dir
assert os.path.exists(os.path.join(cache_dir, "9b5ad71b2ce5302211f9c61530b329a4922fc6a4")) # cl100k_base.tiktoken
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

BROWSER_HEADERS = {
	"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36",
	"Accept": "*/*",
	"Accept-Language": "zh-CN,zh;q=0.9",
	"X-FE-Version": "prod-fe-1.0.76",
	"sec-ch-ua": '"Not;A=Brand";v="99", "Edge";v="139"',
	"sec-ch-ua-mobile": "?0",
	"sec-ch-ua-platform": '"Windows"',
	"Origin": BASE,
}

# 日志
logging.basicConfig(level=logging.DEBUG if DEBUG_MODE else logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

def debug(msg, *args):
	if DEBUG_MODE: log.debug(msg, *args)

# Flask 应用
app = Flask(__name__)

phaseBak = "thinking"
# 工具函数
class utils:
	@staticmethod
	class request:
		@staticmethod
		def chat(data, chat_id):
			debug("收到请求: %s", json.dumps(data))
			url = f"{BASE}/api/chat/completions"
			headers = {
				**BROWSER_HEADERS, 
				"Authorization": f"Bearer {utils.request.token()}", 
				"Referer": f"{BASE}/c/{chat_id}",
				"x-fe-version": "prod-fe-1.0.272", # 使用新版前端版本号
				"content-type": "application/json"
			}
			return requests.post(url, json=data, headers=headers, stream=True, timeout=60)
		@staticmethod
		def image(data_url, chat_id):
			try:
				if ANONYMOUS_MODE or not data_url.startswith("data:"):
					return None

				header, encoded = data_url.split(",", 1)
				mime_type = header.split(";")[0].split(":")[1] if ":" in header else "image/jpeg"

				image_data = base64.b64decode(encoded) # 解码数据
				filename = str(uuid.uuid4())

				debug("上传文件：%s", filename)
				response = requests.post(f"{BASE}/api/v1/files/", files={"file": (filename, image_data, mime_type)}, headers={**BROWSER_HEADERS, "Authorization": f"Bearer {utils.request.token()}", "Referer": f"{BASE}/c/{chat_id}"}, timeout=30)

				if response.status_code == 200:
					result = response.json()
					return f"{result.get('id')}_{result.get('filename')}"
				else:
					raise Exception(response.text)
			except Exception as e:
				debug("图片上传失败: %s", e)
			return None
		@staticmethod
		def id(prefix = "msg") -> str:
			return f"{prefix}-{int(datetime.now().timestamp()*1e9)}"
		@staticmethod
		def token() -> str:
			if not ANONYMOUS_MODE: return TOKEN
			try:
				r = requests.get(f"{BASE}/api/v1/auths/", headers=BROWSER_HEADERS, timeout=8)
				token = r.json().get("token")
				if token:
					debug("获取匿名令牌: %s...", token[:15])
					return token
			except Exception as e:
				debug("匿名令牌获取失败: %s", e)
			return TOKEN
		@staticmethod
		def response(resp):
			resp.headers.update({
				"Access-Control-Allow-Origin": "*",
				"Access-Control-Allow-Methods": "GET, POST, OPTIONS",
				"Access-Control-Allow-Headers": "Content-Type, Authorization",
			})
			return resp
	@staticmethod
	class response:
		@staticmethod
		def parse(stream):
			for line in stream.iter_lines():
				if not line or not line.startswith(b"data: "): continue
				try: 
					line_str = line[6:].decode("utf-8", "ignore")
					print("[DEBUG RAW DATA] " + line_str[:200]) # 强制截断打印看看原始数据长啥样
					line_data = json.loads(line_str)
					# 如果是官网最新的包装格式，把里面的 data 剥出来
					if isinstance(line_data, dict) and "data" in line_data and line_data.get("type") == "chat:completion":
						yield line_data.get("data")
					else:
						yield line_data
				except Exception as e: 
					print(f"[DEBUG EXCEPTION] 解析错误 {e}")
					continue
		@staticmethod
		def format(data):
			print(f"[DEBUG FORMAT IN] phase: {data.get('phase')}, is_done: {data.get('done')}, content length: {len(data.get('delta_content') or '')}")
			if not data: return None
			phase = data.get("phase", "other")
			# 适配 delta_content 或者 edit_content
			content = data.get("delta_content") or data.get("edit_content") or ""
			if not content and phase != "thinking" and phase != "answer": 
				return None
			contentBak = content
			global phaseBak
			if phase == "thinking" or (phase == "answer" and "summary>" in content):
				content = re.sub(r"(?s)<details[^>]*?>.*?</details>", "", content)
				content = content.replace("</thinking>", "").replace("<Full>", "").replace("</Full>", "")

				if phase == "thinking":
					content = re.sub(r'\n*<summary>.*?</summary>\n*', '\n\n', content)

				# 以 <reasoning> 为基底
				content = re.sub(r"<details[^>]*>\n*", "<reasoning>\n\n", content)
				content = re.sub(r"\n*</details>", "\n\n</reasoning>", content)

				if phase == "answer":
					match = re.search(r"(?s)^(.*?</reasoning>)(.*)$", content) # 判断 </reasoning> 后是否有内容
					if match:
						before, after = match.groups()
						if after.strip():
							# </reasoning> 后有内容
							if phaseBak == "thinking":
								# 思考休止 → 结束思考，加上回答
								lstripped_after = after.lstrip('\n')
								content = f"\n\n</reasoning>\n\n{lstripped_after}"
							elif phaseBak == "answer":
								# 回答休止 → 清除所有
								content = ""
						else:
							# 思考休止 → </reasoning> 后无内容
							content = "\n\n</reasoning>"

				if THINK_TAGS_MODE == "reasoning":
					if phase == "thinking": content = re.sub(r'\n>\s?', '\n', content)
					content = re.sub(r'\n*<summary>.*?</summary>\n*', '', content)
					content = re.sub(r"<reasoning>\n*", "", content)
					content = re.sub(r"\n*</reasoning>", "", content)
				elif THINK_TAGS_MODE == "think":
					if phase == "thinking": content = re.sub(r'\n>\s?', '\n', content)
					content = re.sub(r'\n*<summary>.*?</summary>\n*', '', content)
					content = re.sub(r"<reasoning>", "<think>", content)
					content = re.sub(r"</reasoning>", "</think>", content)
				elif THINK_TAGS_MODE == "strip":
					content = re.sub(r'\n*<summary>.*?</summary>\n*', '', content)
					content = re.sub(r"<reasoning>\n*", "", content)
					content = re.sub(r"</reasoning>", "", content)
				elif THINK_TAGS_MODE == "details":
					if phase == "thinking": content = re.sub(r'\n>\s?', '\n', content)
					content = re.sub(r"<reasoning>", "<details type=\"reasoning\" open><div>", content)
					thoughts = ""
					if phase == "answer":
						# 判断是否有 <summary> 内容
						summary_match = re.search(r"(?s)<summary>.*?</summary>", before)
						duration_match = re.search(r'duration="(\d+)"', before)
						if summary_match:
							# 有内容 → 直接照搬
							thoughts = f"\n\n{summary_match.group()}"
						# 判断是否有 duration 内容
						elif duration_match:
							# 有内容 → 通过 duration 生成 <summary>
							thoughts = f'\n\n<summary>Thought for {duration_match.group(1)} seconds</summary>'
					content = re.sub(r"</reasoning>", f"</div>{thoughts}</details>", content)
				else:
					content = re.sub(r"</reasoning>", "</reasoning>\n\n", content)
					debug("警告：THINK_TAGS_MODE 传入了未知的替换模式，将使用 <reasoning> 标签。")

			phaseBak = phase
			if repr(content) != repr(contentBak):
				debug("R 内容: %s %s", phase, repr(contentBak))
				debug("W 内容: %s %s", phase, repr(content))

			# print(f"[DEBUG FORMAT OUT] phase: {phase}, content ready to send: {repr(content)[:50]}")

			if phase == "thinking" and THINK_TAGS_MODE == "reasoning":
				return {"role": "assistant", "reasoning_content": content}
			elif repr(content):
				return {"role": "assistant", "content":content}
			else:
				return None
		@staticmethod
		def count(text):
			return len(enc.encode(text))

# 路由
@app.route("/v1/models", methods=["GET", "POST", "OPTIONS"])
def models():
	if request.method == "OPTIONS": return utils.request.response(make_response())
	try:
		def format_model_name(name: str) -> str:
			"""格式化模型名:
			- 单段: 全大写
			- 多段: 第一段全大写, 后续段首字母大写
			- 数字保持不变, 符号原样保留
			"""
			if not name: return ""
			parts = name.split('-')
			if len(parts) == 1:
				return parts[0].upper()
			formatted = [parts[0].upper()]
			for p in parts[1:]:
				if not p:
					formatted.append("")
				elif p.isdigit():
					formatted.append(p)
				elif any(c.isalpha() for c in p):
					formatted.append(p.capitalize())
				else:
					formatted.append(p)
			return "-".join(formatted)

		def is_english_letter(ch: str) -> bool:
			"""判断是否是英文字符 (A-Z / a-z)"""
			return 'A' <= ch <= 'Z' or 'a' <= ch <= 'z'

		headers = {**BROWSER_HEADERS, "Authorization": f"Bearer {utils.request.token()}"}
		r = requests.get(f"{BASE}/api/models", headers=headers, timeout=8).json()
		models = []
		for m in r.get("data", []):
			if not m.get("info", {}).get("is_active", True):
				continue
			model_id, model_name = m.get("id"), m.get("name")
			if model_id.startswith(("GLM", "Z")):
				model_name = model_id
			if not model_name or not is_english_letter(model_name[0]):
				model_name = format_model_name(model_id)
			models.append({
				"id": model_id,
				"object": "model",
				"name": model_name,
				"created": m.get("info", {}).get("created_at", int(datetime.now().timestamp())),
				"owned_by": "z.ai"
			})
		return utils.request.response(jsonify({"object":"list","data":models}))
	except Exception as e:
		debug("模型列表失败: %s", e)
		return utils.request.response(jsonify({"error":"fetch models failed"})), 500

@app.route("/v1/chat/completions", methods=["GET", "POST", "OPTIONS"])
def OpenAI_Compatible():
	if request.method == "OPTIONS": return utils.request.response(make_response())
	odata = request.get_json(force=True, silent=True) or {}

	id = utils.request.id("chat")
	model = odata.get("model", MODEL)
	messages = odata.get("messages", [])
	features = odata.get("features", { "enable_thinking": True })
	stream = odata.get("stream", False)
	include_usage = stream and odata.get("stream_options", {}).get("include_usage", False)

	for message in messages:
		if isinstance(message.get("content"), list):
			for content_item in message["content"]:
				if content_item.get("type") == "image_url":
					url = content_item.get("image_url", {}).get("url", "")
					if url.startswith("data:"):
						file_url = utils.request.image(url, id) # 上传图片
						if file_url:
							content_item["image_url"]["url"] = file_url # 上传后的图片链接

	# 提取最后一条提问的文字，供 signature_prompt 使用
	last_user_message = ""
	for m in reversed(messages):
		if m.get("role") == "user":
			content_data = m.get("content", "")
			if isinstance(content_data, str):
				last_user_message = content_data
			elif isinstance(content_data, list):
				for item in content_data:
					if item.get("type") == "text":
						last_user_message += item.get("text", "")
			break

	full_features = {
		"image_generation": False,
		"web_search": False,
		"auto_web_search": False,
		"preview_mode": True,
		"flags": [],
		"enable_thinking": True,
		**features
	}

	data = {
		**odata, 
		"stream": True,
		"model": model,
		"messages": messages,
		"signature_prompt": last_user_message,
		"params": {},
		"extra": {},
		"mcp_servers": ["vlm-image-search", "vlm-image-recognition", "vlm-image-processing"],
		"features": full_features,
		"variables": {
			"{{USER_NAME}}": "User",
			"{{USER_LOCATION}}": "Unknown",
			"{{CURRENT_DATETIME}}": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
			"{{CURRENT_DATE}}": datetime.now().strftime("%Y-%m-%d"),
			"{{CURRENT_TIME}}": datetime.now().strftime("%H:%M:%S"),
			"{{CURRENT_WEEKDAY}}": datetime.now().strftime("%A"),
			"{{CURRENT_TIMEZONE}}": "Asia/Shanghai",
			"{{USER_LANGUAGE}}": "zh-CN"
		},
		"chat_id": id,
		"id": utils.request.id(),
		"current_user_message_id": str(uuid.uuid4()),
		"current_user_message_parent_id": None,
		"background_tasks": {
			"title_generation": True,
			"tags_generation": True
		}
	}

	try:
		response = utils.request.chat(data, id)
		print(f"\n======================================")
		print(f"[HTTP 请求完成] 目标: /api/v2/chat/completions | 状态码: {response.status_code}")
		
		# 如果上游拒绝了（比如 403 / 400 等），我们强制打印真实原因，并且直接终止，不再去读流。
		if response.status_code != 200:
			error_text = response.text
			print(f"[重点关注！HTTP ERROR] Z.ai 上游拒绝请求，返回原文:\n{error_text}")
			print(f"======================================\n")
			return utils.request.response(make_response(f"上游接口报错 (HTTP {response.status_code}): {error_text}", 502))
			
	except Exception as e:
		print(f"[严重错误] 请求完全失败: {e}")
		return utils.request.response(make_response(f"上游请求失败: {e}", 502))

	prompt_tokens = utils.response.count("".join(
		c if isinstance(c, str) else (c.get("text", "") if isinstance(c, dict) and c.get("type") == "text" else "")
		for m in messages
		for c in ([m["content"]] if isinstance(m.get("content"), str) else (m.get("content") or []))
	))
	if stream:
		def stream():
			completion_str = ""
			completion_tokens = 0

			# 处理流式响应数据
			for data in utils.response.parse(response):
				is_done = data.get("data", {}).get("done", False)
				delta = utils.response.format(data)
				finish_reason = "stop" if is_done else None

				if delta:
					chunk = {
						"id": utils.request.id('chatcmpl'),
						"object": "chat.completion.chunk",
						"created": int(datetime.now().timestamp()),
						"model": model,
						"choices": [
							{
								"index": 0,
								"delta": delta,
								"message": delta,
								"finish_reason": finish_reason
							}
						]
					}
					yield f"data: {json.dumps(chunk)}\n\n"

					# 累积实际生成的内容
					if "content" in delta:
						completion_str += delta["content"]
					if "reasoning_content" in delta:
						completion_str += delta["reasoning_content"]
					completion_tokens = utils.response.count(completion_str) # 计算 tokens
				if is_done:
					done_chunk = {
						'id': utils.request.id('chatcmpl'),
						'object': 'chat.completion.chunk',
						'created': int(datetime.now().timestamp()),
						'model': model,
						'choices': [
							{
								'index': 0,
								'delta': {"role": "assistant"},
								'message': {"role": "assistant"},
								'finish_reason': "stop"
							}
						]
					}
					yield f"data: {json.dumps(done_chunk)}\n\n"
					break

			if include_usage:
				# 发送 usage 统计信息
				usage_chunk = {
					"id": utils.request.id('chatcmpl'),
					"object": "chat.completion.chunk",
					"created": int(datetime.now().timestamp()),
					"model": model,
					"choices": [],
					"usage": {
						"prompt_tokens": prompt_tokens,
						"completion_tokens": completion_tokens,
						"total_tokens": prompt_tokens + completion_tokens
					}
				}
				yield f"data: {json.dumps(usage_chunk)}\n\n"

			# 发送 [DONE] 标志，表示流结束
			yield "data: [DONE]\n\n"

		# 返回 Flask 的流式响应
		return Response(stream(), mimetype="text/event-stream")
	else:
		# 上游不支持非流式，所以先用流式获取所有内容
		contents = {
			"content": [],
			"reasoning_content": []
		}
		for odata in utils.response.parse(response):
			if odata.get("data", {}).get("done"):
				break
			delta = utils.response.format(odata)
			if delta:
				if "content" in delta:
					contents["content"].append(delta["content"])
				if "reasoning_content" in delta:
					contents["reasoning_content"].append(delta["reasoning_content"])

		# 构建最终消息内容
		final_message = {"role": "assistant"}
		completion_str = ""
		if contents["reasoning_content"]:
			final_message["reasoning_content"] = "".join(contents["reasoning_content"])
			completion_str += "".join(contents["reasoning_content"])
		if contents["content"]:
			final_message["content"] = "".join(contents["content"])
			completion_str += "".join(contents["content"])
		completion_tokens = utils.response.count(completion_str) # 计算 tokens

		# 返回 Flask 响应
		return utils.request.response(jsonify({
			"id": utils.request.id("chatcmpl"),
			"object": "chat.completion",
			"created": int(datetime.now().timestamp()),
			"model": model,
			"choices": [{
				"index": 0,
				"delta": final_message,
				"message": final_message,
				"finish_reason": "stop"
			}],
			"usage": {
				"prompt_tokens": prompt_tokens,
				"completion_tokens": completion_tokens,
				"total_tokens": prompt_tokens + completion_tokens
			}
		}))

# 主入口
if __name__ == "__main__":
	log.info("---------------------------------------------------------------------")
	log.info("Z.ai 2 API")
	log.info("将 Z.ai 代理为 OpenAI Compatible 格式")
	log.info("基于 https://github.com/kbykb/OpenAI-Compatible-API-Proxy-for-Z 重构")
	log.info("---------------------------------------------------------------------")
	log.info("服务端口：%s", PORT)
	log.info("备选模型：%s", MODEL)
	log.info("思考处理：%s", THINK_TAGS_MODE)
	log.info("访客模式：%s", ANONYMOUS_MODE)
	log.info("显示调试：%s", DEBUG_MODE)
	app.run(host="0.0.0.0", port=PORT, threaded=True, debug=DEBUG_MODE)