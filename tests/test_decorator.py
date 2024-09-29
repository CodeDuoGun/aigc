import requests
def retry(max_retries=3):    
    def decorator(func):        
        def wrapper(*args, **kwargs):            
            for attempt in range(max_retries):                
                try:                    
                    return func(*args, **kwargs)                
                except Exception as e:                    
                    if attempt < max_retries - 1:                        
                        print(f"Attempt {attempt + 1} failed, retrying...")                    
                    else:                        
                        print(f"Max retries ({max_retries}) reached, last error: {e}")                        
                        return None        
        return wrapper    
    return decorator

@retry(max_retries=3)
def test_api_with_retry(url):    
    response = requests.get(url)    
    if response.status_code == 200:        
        print("Request successful")    
    else:        
        raise Exception(f"Request failed with status code {response.status_code}")
test_api_with_retry("https://example.com/api")