import requests
import json
import time

URL = 'http://localhost:8000/api/chat'
HEADERS = {'Content-Type': 'application/json'}

tests = [
    {
        'name': '1. General Medical',
        'query': 'What is hypertension?',
        'expected_tool': None,
        'expected_in_response': ['hypertension', 'blood']
    },
    {
        'name': '2. Non-Medical',
        'query': 'What is the capital of France?',
        'expected_tool': None,
        'expected_in_response': ['only help']
    },
    {
        'name': '3. Symptoms - English',
        'query': 'I have a severe headache and a very stiff neck, maybe a fever.',
        'expected_tool': 'symptoms_analysis',
        'expected_in_response': ['meningitis']
    },
    {
        'name': '4. Symptoms - Arabic',
        'query': 'انا عندى صداع جامد جدا ورقبتى وجعانى اوى وممكن سخونيه',
        'expected_tool': 'symptoms_analysis',
        'expected_in_response': ['name', 'rationale']
    },
    {
        'name': '5. Lab Explanations',
        'query': 'My HbA1c is 8.5%, what does that mean?',
        'expected_tool': 'lab_report_explanation',
    },
    {
        'name': '6. Drug Interactions',
        'query': 'Can I take ibuprofen and warfarin together safely?',
        'expected_tool': 'drug_interaction_checker',
    }
]

print('\n' + '='*50)
print('FULL MEDICAL AGENT DIAGNOSTIC SUITE')
print('='*50 + '\n')

passed = 0
failed = 0

for t in tests:
    print(f'Testing: {t["name"]}')
    print(f'Query: "{t["query"]}"')
    try:
        start_time = time.time()
        res = requests.post(URL, json={'query': t['query']}, headers=HEADERS, timeout=60)
        elapsed = time.time() - start_time
        
        if res.status_code == 200:
            data = res.json()
            tool_used = data.get('tool_used')
            response_text = data.get('response', '')
            
            tool_match = (tool_used == t['expected_tool'])
            content_match = True
            if 'expected_in_response' in t:
                for keyword in t['expected_in_response']:
                    if keyword.lower() not in response_text.lower():
                        content_match = False
                        print(f'   [X] Missing expected keyword: {keyword}')
                        
            if tool_match and content_match:
                print(f'   [V] PASS ({elapsed:.1f}s)')
                passed += 1
            else:
                print(f'   [X] FAIL ({elapsed:.1f}s)')
                print(f'      Expected Tool: {t["expected_tool"]} | Actual Tool: {tool_used}')
                print(f'      Response Snippet: {response_text[:150]}...')
                failed += 1
        else:
            print(f'   [X] FAIL HTTP {res.status_code}')
            print('      ' + res.text[:200])
            failed += 1
    except Exception as e:
        print(f'   [X] FAIL Exception: {e}')
        failed += 1
    
    print('-'*50)

print(f'\nDIAGNOSTIC COMPLETE | {passed}/{passed+failed} TESTS PASSED\n')
