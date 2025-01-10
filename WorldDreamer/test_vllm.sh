curl -X POST "http://172.31.208.10:10019/generate" \
     -H "Content-Type: application/json" \
     -d '{
           "prompt": "Describe the image:",
           "max_tokens": 100,
           "temperature": 0.7
         }'
