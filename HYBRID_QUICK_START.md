# 🚀 PCIe Debug Agent - Hybrid LLM Quick Start

## ✅ Setup Complete!

Your hybrid LLM system is now integrated and ready to use.

## 🎯 Quick Commands

### 1. Check Model Status
```bash
./pcie-debug-hybrid model status
```

### 2. Run Quick Analysis (Llama)
```bash
./pcie-debug-hybrid analyze -q "What is a PCIe TLP error?" -t quick
```

### 3. Run Detailed Analysis (DeepSeek)
```bash
./pcie-debug-hybrid analyze -q "Analyze PCIe link training failure" -t detailed
```

### 4. Auto Mode (Intelligent Selection)
```bash
./pcie-debug-hybrid analyze -q "Debug intermittent device disconnection" -t auto
```

### 5. Launch Web UI
```bash
./run_hybrid_ui.sh
```
Then open: http://localhost:8501

## 📊 Model Comparison

| Feature | Llama 3.2 3B | DeepSeek Q4_1 |
|---------|--------------|---------------|
| Speed | ~25s ⚡ | 90s+ 🐌 |
| Memory | 1.8GB 💚 | 5.2GB 🟡 |
| Quality | Good ✅ | Excellent 🌟 |
| Best For | Interactive | Deep Analysis |

## 🔧 Configuration

Settings are in: `configs/settings.yaml`

To use hybrid provider, set:
```yaml
llm:
  provider: hybrid
  model: auto-hybrid
```

## 📝 Examples

### Quick PCIe Error Check
```bash
echo "[10:15:30] PCIe: Link training failed" | ./pcie-debug-hybrid analyze -q "What's wrong?" -t quick
```

### Detailed Root Cause Analysis
```bash
./pcie-debug-hybrid analyze error.log -q "Provide comprehensive root cause analysis" -t detailed
```

## 🎉 Enjoy your hybrid AI-powered PCIe debugging!
