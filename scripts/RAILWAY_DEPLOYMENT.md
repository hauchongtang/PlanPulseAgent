# Railway Deployment Guide for PlanPulseAgent

## 🚀 Railway Configuration

Your `railway.json` is optimized for the supervisor pattern architecture with:

### Deployment Settings
- **Server**: Gunicorn with Uvicorn workers (production-ready)
- **Workers**: 2 workers (optimal for Railway's resource limits)
- **Timeout**: 600 seconds (handles complex agent workflows)
- **Health Check**: `/health` endpoint monitoring
- **Auto-restart**: On failure with 3 retry attempts

## 📋 Pre-Deployment Checklist

### ✅ Required Environment Variables
Set these in your Railway project settings:

```bash
GOOGLE_API_KEY=your_google_api_key_here
NOTION_TOKEN=your_notion_integration_token
NOTION_DATABASE_ID=your_notion_database_id
```

### ✅ Files Ready for Deployment
- [x] `railway.json` - Railway configuration
- [x] `requirements.txt` - Python dependencies
- [x] `main.py` - FastAPI application entry point
- [x] `.gitignore` - Excludes sensitive files
- [x] Supervisor pattern architecture

## 🔧 Railway Deployment Steps

### 1. Connect Repository
```bash
# Option A: GitHub Integration (Recommended)
1. Connect your GitHub repository to Railway
2. Railway will auto-detect Python and use railway.json

# Option B: Railway CLI
railway login
railway link
railway up
```

### 2. Set Environment Variables
In Railway Dashboard:
1. Go to your project
2. Click "Variables" tab
3. Add the required environment variables:
   - `GOOGLE_API_KEY`
   - `NOTION_TOKEN` 
   - `NOTION_DATABASE_ID`

### 3. Deploy
Railway will automatically:
1. Install dependencies from `requirements.txt`
2. Use the optimized start command from `railway.json`
3. Enable health checks on `/health`
4. Provide a public URL

## 🔍 Monitoring Your Deployment

### Health Check Endpoints
- **Service Health**: `https://your-app.railway.app/health`
- **Agent Status**: `https://your-app.railway.app/v1/telegram/health`
- **API Docs**: `https://your-app.railway.app/docs`

### Agent Information
- **Agent Capabilities**: `https://your-app.railway.app/v1/telegram/agents`
- **Available Tools**: `https://your-app.railway.app/v1/telegram/tools`

## 🧪 Testing Your Deployment

### Test Agent Routing
```bash
# Test Math Agent
curl -X POST "https://your-app.railway.app/v1/telegram/" \
  -H "Content-Type: application/json" \
  -d '{"chat_id": "test", "chat_message": "Calculate 15% of 200"}'

# Test Notion Agent  
curl -X POST "https://your-app.railway.app/v1/telegram/" \
  -H "Content-Type: application/json" \
  -d '{"chat_id": "test", "chat_message": "Show my events for tomorrow"}'
```

### Expected Response Format
```json
{
  "chat_id": "test",
  "chat_message": "Calculate 15% of 200", 
  "response": "15% of 200 is 30",
  "success": true,
  "metadata": {},
  "supervisor_metadata": {
    "selected_agent": "math_agent",
    "confidence_scores": {
      "math_agent": 0.9,
      "notion_agent": 0.1
    }
  }
}
```

## ⚡ Performance Optimization

### Railway-Specific Optimizations
- **2 Workers**: Balanced for Railway's CPU limits
- **Preload**: Faster cold starts
- **Max Requests**: Automatic worker recycling
- **Keep Alive**: Efficient connection handling

### Agent-Specific Optimizations
- **Lazy Loading**: Agents created on-demand
- **Confidence Caching**: Faster routing decisions
- **Error Resilience**: Graceful fallbacks

## 🐛 Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Check logs in Railway dashboard
# Ensure all dependencies in requirements.txt
```

#### 2. API Key Issues
```bash
# Verify environment variables are set
# Check Railway dashboard > Variables tab
```

#### 3. Health Check Failures
```bash
# Verify /health endpoint responds
# Check agent initialization in logs
```

#### 4. Agent Routing Issues
```bash
# Check supervisor metadata in responses
# Verify confidence scoring logic
```

### Railway Logs
Monitor deployment logs in Railway dashboard:
- Build logs for dependency issues
- Runtime logs for agent errors
- Health check status

## 🔐 Security Notes

- Environment variables are encrypted in Railway
- `.gitignore` prevents sensitive files from being deployed
- API keys are not logged or exposed
- Health checks don't expose sensitive information

## 📈 Scaling Considerations

### Current Configuration
- **Workers**: 2 (suitable for most traffic)
- **Memory**: Railway auto-scales
- **CPU**: Optimized for supervisor pattern

### Future Scaling
- Increase workers in `railway.json` if needed
- Monitor performance via Railway metrics
- Consider Redis for agent state caching

Your supervisor pattern application is now ready for Railway deployment! 🚀
