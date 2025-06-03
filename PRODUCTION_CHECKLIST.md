# Enhanced RAG v3 Production Deployment Checklist

## 🚀 Pre-Deployment Checklist

### ✅ **System Requirements**
- [ ] Python 3.8+ installed
- [ ] Minimum 4GB RAM available
- [ ] 2GB disk space for vector databases
- [ ] Network access for model downloads (if using cloud models)

### ✅ **Dependencies**
- [ ] Install core dependencies: `pip install numpy>=1.21.0 sentence-transformers>=2.2.0 rank-bm25>=0.2.2`
- [ ] Verify dependencies: `python -c "import numpy, sentence_transformers, rank_bm25; print('✅ All dependencies available')"`
- [ ] Test graceful fallback: Start system without dependencies and verify basic RAG works

### ✅ **Configuration**
- [ ] Review and customize `configs/rag_v3.yaml` configuration
- [ ] Set appropriate confidence thresholds for your use case
- [ ] Configure cache size based on available memory
- [ ] Enable/disable features based on requirements

### ✅ **Vector Database**
- [ ] Build vector database: `pcie-debug vectordb build`
- [ ] Verify database size and document count
- [ ] Test search functionality: `/search test query`
- [ ] Backup vector database files

### ✅ **Testing**
- [ ] Run integration tests: `python focused_integration_test.py`
- [ ] Test all new CLI commands: `/rag_analyze`, `/rag_verify`, `/rag_v3_status`, `/suggest`
- [ ] Verify fallback mechanisms work correctly
- [ ] Test with various query types and edge cases

## 🔧 **Deployment Steps**

### 1. **Install and Configure**
```bash
# Clone/update repository
git pull origin main

# Install dependencies
pip install -r requirements.txt
pip install numpy>=1.21.0 sentence-transformers>=2.2.0 rank-bm25>=0.2.2

# Verify installation
python focused_integration_test.py
```

### 2. **Initialize Vector Database**
```bash
# Build vector database with current embedding model
pcie-debug vectordb build

# Verify database
pcie-debug vectordb status
```

### 3. **Test Enhanced RAG v3**
```bash
# Start interactive shell
python src/cli/main.py

# Test v3 status
> /rag_v3_status

# Test enhanced analysis
> /rag_analyze What are PCIe LTSSM states?

# Test verification
> /rag_verify How does PCIe flow control work?

# Test suggestions
> /suggest ltssm power
```

### 4. **Configure for Production**
```bash
# Copy default config
cp configs/settings.yaml configs/settings_prod.yaml

# Edit production settings
# - Set appropriate model limits
# - Configure logging levels  
# - Set cache sizes
# - Enable/disable features
```

## 📊 **Performance Monitoring**

### **Key Metrics to Monitor**
- [ ] Average response time (`/rag_v3_status` → metrics)
- [ ] Confidence score distribution
- [ ] Cache hit rates
- [ ] Memory usage patterns
- [ ] Error rates and types

### **Monitoring Commands**
```bash
# Check system status
> /rag_v3_status

# Monitor performance
> /cost

# Check memory usage
> /doctor

# View detailed metrics
> /rag_status
```

### **Performance Baselines**
- **Response Time**: < 2 seconds for typical queries
- **Confidence**: > 70% for domain-specific queries  
- **Cache Hit Rate**: > 30% for repeated query patterns
- **Memory Growth**: < 100MB per 1000 queries

## 🛡️ **Security Checklist**

### **Input Validation**
- [ ] Query length limits enforced (max 1000 chars)
- [ ] Concept count limits enforced (max 10)
- [ ] Special character handling verified
- [ ] SQL injection patterns rejected

### **Access Control**
- [ ] CLI access appropriately restricted
- [ ] File system permissions configured
- [ ] Network access limited to required services
- [ ] Logging configured for security events

### **Data Protection**
- [ ] No sensitive data in query logs
- [ ] Vector database access secured
- [ ] Configuration files protected
- [ ] Backup procedures established

## 🔄 **Rollback Plan**

### **Fallback Mechanisms**
1. **Automatic Fallback**: System automatically falls back to basic RAG if v3 fails
2. **Configuration Rollback**: Reset to basic configuration: `/rag off` then `/rag on`
3. **Dependency Rollback**: Remove v3 dependencies to force basic mode
4. **Code Rollback**: Revert to previous git commit if needed

### **Rollback Commands**
```bash
# Disable Enhanced RAG v3
> /rag off

# Check system still works
> What are PCIe errors?

# Re-enable basic RAG
> /rag on

# Verify basic functionality
> /rag_status
```

## 📋 **Post-Deployment Validation**

### **Functional Tests**
- [ ] Test basic query processing
- [ ] Test enhanced v3 analysis (`/rag_analyze`)
- [ ] Test answer verification (`/rag_verify`)
- [ ] Test question suggestions (`/suggest`)
- [ ] Test system status (`/rag_v3_status`)

### **Performance Tests**
- [ ] Response time within acceptable limits
- [ ] Memory usage stable over time
- [ ] Cache performance effective
- [ ] No memory leaks detected

### **Error Handling Tests**
- [ ] Invalid inputs handled gracefully
- [ ] Network failures don't crash system
- [ ] Dependency failures trigger fallback
- [ ] Edge cases handled appropriately

## 🚨 **Troubleshooting Guide**

### **Common Issues**

#### **"Enhanced RAG v3 not available"**
```bash
# Check dependencies
python -c "import numpy, sentence_transformers, rank_bm25"

# If missing, install
pip install numpy sentence-transformers rank-bm25

# Restart shell
```

#### **"Vector database not loaded"**
```bash
# Build database
pcie-debug vectordb build

# Check status
pcie-debug vectordb status

# Verify permissions
ls -la data/vectorstore/
```

#### **Slow Performance**
```bash
# Check system resources
> /doctor

# Review cache settings
> /rag_v3_status

# Consider reducing max_results in config
```

#### **Low Confidence Scores**
```bash
# Check vector database quality
> /rag_files

# Review confidence thresholds
> /rag_v3_status

# Consider rebuilding with better embedding model
```

## 📞 **Support and Maintenance**

### **Regular Maintenance**
- [ ] Monitor system performance weekly
- [ ] Update dependencies monthly  
- [ ] Rebuild vector database when adding new knowledge
- [ ] Review and tune configuration quarterly

### **Support Contacts**
- **Technical Issues**: Check logs in `logs/` directory
- **Performance Issues**: Use `/doctor` and `/rag_v3_status` 
- **Configuration Help**: Review `DEPENDENCIES_V3.md` and config files

### **Documentation**
- [ ] `README.md` - General usage
- [ ] `DEPENDENCIES_V3.md` - Dependency management
- [ ] `INTEGRATION_STATUS.md` - Integration details
- [ ] This checklist for deployment

## ✅ **Sign-off**

**Deployment completed by**: ________________  
**Date**: ________________  
**Version**: Enhanced RAG v3  
**Environment**: ________________  

**Validation Results**:
- [ ] All tests passed
- [ ] Performance acceptable  
- [ ] Security reviewed
- [ ] Rollback plan verified
- [ ] Documentation updated

**Notes**: ________________________________________________