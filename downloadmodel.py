import os
import sys
from transformers import TRANSFORMERS_CACHE, AutoTokenizer, AutoModelForSeq2SeqLM

# 模型的名称
MODEL_NAME = "Salesforce/codet5-base"
MIRROR_ENDPOINT = "https://hf-mirror.com"
ORIGINAL_ENDPOINT = "https://huggingface.co"

# --- 核心修改：设置 Hugging Face 镜像 ---
os.environ["HF_ENDPOINT"] = MIRROR_ENDPOINT


# =======================================================
# 诊断函数：使用 Python 内置库测试连接
# =======================================================

def run_connection_test(url, timeout=5):
    """尝试使用 Python 标准库测试连接"""
    try:
        import urllib.request
        print(f"   -> Testing: {url}...")
        # 尝试发起 HEAD 请求
        req = urllib.request.Request(url, method='HEAD')
        with urllib.request.urlopen(req, timeout=timeout) as response:
            status = response.getcode()
            if status in [200, 307]:
                return True, f"Success (Status {status})"
            else:
                return False, f"HTTP Error (Status {status})"
    except ImportError:
        # Fallback if urllib.request is not available (unlikely)
        return False, "urllib.request not found."
    except Exception as e:
        return False, f"Connection Failed: {type(e).__name__} - {e}"

# =======================================================
# 主执行流程
# =======================================================

print(f"Transformers cache directory: {TRANSFORMERS_CACHE}")
print(f"Loading model: {MODEL_NAME} via mirror: {os.environ['HF_ENDPOINT']}...")
print("-" * 40)


try:
    # 官方下载方法，通过 HF_ENDPOINT 指向的镜像下载
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    print("\n✅ 模型和分词器加载成功！")
    print("Tokenizer type:", type(tokenizer))
    print("Model type:", type(model))

except Exception as e:
    # =======================================================
    # 失败诊断部分
    # =======================================================
    print(f"\n❌ 模型加载失败，正在进行详细诊断...")
    print(f"详细错误: {e}")
    print("\n--- Python 环境网络诊断 ---")
    
    # 1. 检查代理设置
    print(f"1. Proxy Check (from os.environ):")
    print(f"   - HTTP_PROXY: {os.environ.get('HTTP_PROXY', 'Not set')}")
    print(f"   - HTTPS_PROXY: {os.environ.get('HTTPS_PROXY', 'Not set')}")

    # 2. 尝试连接镜像源
    test_mirror_url = f"{MIRROR_ENDPOINT}/Salesforce/codet5-base/resolve/main/config.json"
    is_mirror_ok, mirror_result = run_connection_test(test_mirror_url)
    print(f"\n2. Mirror Connection ({MIRROR_ENDPOINT}):")
    print(f"   - Result: {'SUCCESS' if is_mirror_ok else 'FAILED'}")
    print(f"   - Detail: {mirror_result}")

    # 3. 尝试连接原始源 (用于对比，看是否网络完全不通)
    test_original_url = f"{ORIGINAL_ENDPOINT}/Salesforce/codet5-base/resolve/main/config.json"
    is_original_ok, original_result = run_connection_test(test_original_url)
    print(f"\n3. Original Connection ({ORIGINAL_ENDPOINT}):")
    print(f"   - Result: {'SUCCESS' if is_original_ok else 'FAILED'}")
    print(f"   - Detail: {original_result}")

    # 4. 根据诊断结果给出建议
    print("\n--- 诊断建议 ---")
    if not is_mirror_ok:
        if "TimeoutError" in mirror_result or "ConnectTimeoutError" in mirror_result:
             print("💡 结论: Python 环境连接 **镜像源** 时发生**超时**。")
             print("   - 检查防火墙或网络访问控制列表 (ACL)。")
             print("   - 如果使用了代理，请确保代理设置（HTTP_PROXY/HTTPS_PROXY）在脚本运行前已正确导入。")
        elif "SSLError" in mirror_result:
             print("💡 结论: Python 环境连接 **镜像源** 时发生 **SSL证书错误**。")
             print("   - 尝试升级 `certifi` 库，或确认系统证书完整性。")
        else:
             print("💡 结论: Python 环境无法通过任何方式连接到镜像源。")
             
    if is_original_ok and not is_mirror_ok:
        print("⚠️ 奇怪! 原始 Hugging Face 可通，但镜像不通。尝试移除 HF_ENDPOINT 变量。")

    # 5. 提示本地下载
    print("\n--- 本地加载提示 ---")
    print("💡 无论如何，最可靠的解决方案是 **手动下载** 模型文件后，从本地路径加载。")
    try:
        from transformers.utils import cached_file
        local_config_path = cached_file(MODEL_NAME, "config.json")
        local_model_path = os.path.dirname(local_config_path)
        print(f"🔍 如果文件已部分缓存，本地路径可能在: {local_model_path}")
    except:
        pass

    # 退出程序，避免继续执行
    sys.exit(1)

# import os

# # ✅ 必须放在最前面！确保所有后续库都使用镜像
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# from transformers import AutoTokenizer, AutoModel

# # 查看缓存路径（可选）
# from transformers import TRANSFORMERS_CACHE
# print(f"Transformers cache directory: {TRANSFORMERS_CACHE}")

# # ✅ 使用 AutoTokenizer 和 AutoModel 加载 CodeBERT（不是 RobertaTokenizer/Model）
# model_name = "microsoft/codebert-base"

# print(f"Loading model: {model_name} from {os.environ['HF_ENDPOINT']}")

# # 自动从镜像站下载并缓存
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModel.from_pretrained(model_name, use_safetensors=True) 

# print("✅ 模型和分词器加载成功！")
# print("Tokenizer type:", type(tokenizer))
# print("Model type:", type(model))
# 再加一个看localpath的代码