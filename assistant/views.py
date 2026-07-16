from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .rag_engine import process_uploaded_file, rag_engine, general_chat_engine

SESSION_VECTOR_STORES = {}

# --- PAGE RENDERING VIEW ---
def index(request):
    """Renders the single-page application."""
    if not request.session.session_key:
        request.session.create()
    return render(request, 'assistant/index.html')

# --- API ENDPOINTS ---
@csrf_exempt
def upload_file(request):
    if not request.session.session_key:
        request.session.create()

    if request.method == 'POST' and request.FILES.get('file'):
        session_id = request.session.session_key
        uploaded_file = request.FILES['file']
        
        try:
            vector_store = process_uploaded_file(uploaded_file)
            if not vector_store:
                return JsonResponse({'status': 'error', 'message': 'Could not read text. Is this a scanned PDF?'}, status=400)

            SESSION_VECTOR_STORES[session_id] = vector_store
            request.session['doc_chat_history'] = []
            request.session.modified = True
            return JsonResponse({'status': 'success', 'message': 'Transcript ready!'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)
            
    return JsonResponse({'status': 'error', 'message': 'No file uploaded'}, status=400)

@csrf_exempt
def ask_question(request):
    if not request.session.session_key:
        request.session.create()

    if request.method == 'POST':
        session_id = request.session.session_key
        question = request.POST.get('question', '')
        
        vector_store = SESSION_VECTOR_STORES.get(session_id, [])
        if not vector_store:
            return JsonResponse({'status': 'error', 'message': 'No document found. Please upload a transcript.'}, status=400)
            
        chat_history = request.session.get('doc_chat_history', [])
        history_text = "\n".join([f"User: {msg['q']}\nAI: {msg['a']}" for msg in chat_history[-3:]])
        
        answer = rag_engine(question, vector_store, history_text)
        
        chat_history.append({'q': question, 'a': answer})
        request.session['doc_chat_history'] = chat_history
        request.session.modified = True 
        
        return JsonResponse({'status': 'success', 'answer': answer})
    return JsonResponse({'status': 'error'}, status=400)

@csrf_exempt
def ask_general(request):
    if not request.session.session_key:
        request.session.create()

    if request.method == 'POST':
        question = request.POST.get('question', '')
        
        chat_history = request.session.get('general_chat_history', [])
        history_text = "\n".join([f"User: {msg['q']}\nAI: {msg['a']}" for msg in chat_history[-3:]])
        
        answer = general_chat_engine(question, history_text)
        
        chat_history.append({'q': question, 'a': answer})
        request.session['general_chat_history'] = chat_history
        request.session.modified = True 
        
        return JsonResponse({'status': 'success', 'answer': answer})
    return JsonResponse({'status': 'error'}, status=400)

@csrf_exempt
def clear_history(request):
    if request.method == 'POST':
        chat_type = request.POST.get('type', 'doc')
        key = 'doc_chat_history' if chat_type == 'doc' else 'general_chat_history'
        request.session[key] = []
        request.session.modified = True
        return JsonResponse({'status': 'success'})
    return JsonResponse({'status': 'error'}, status=400)