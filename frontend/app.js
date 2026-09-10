const API = 'https://ml-portfolio-1.onrender.com';

const CODE_EXAMPLES = [
  "def parse_config(file_path:str):\n    config = {}\n    with open(file_path) as f:\n        for line in f:\n            if line.strip() and not line.startswith('#'):\n                ",
  "def retry(func, max_attempts:int, delay:float):\n    attempts = 0\n    while attempts < max_attempts:\n        try:\n            ",
  "def read_csv(file_path:str):\n    with open(file_path) as f:\n        headers = f.readline().strip().split(',')\n        rows = []\n        for line in f:\n            ",
  "def validate_email(email:str) -> bool:\n    if '@' not in email:\n        return False\n    parts = email.split('@')\n    if len(parts) != 2:\n        ",
  "def flatten(nested:list) -> list:\n    result = []\n    for item in nested:\n        if isinstance(item, list):\n            ",
  "def chunk(lst:list, size:int) -> list:\n    result = []\n    for i in range(0, len(lst), size):\n        ",
  "def merge_dicts(base:dict, override:dict) -> dict:\n    result = base.copy()\n    for key, value in override.items():\n        ",
  "def paginate(items:list, page:int, page_size:int) -> list:\n    start = (page - 1) * page_size\n    end = start + page_size\n    ",
  "def deep_get(obj:dict, *keys, default=None):\n    current = obj\n    for key in keys:\n        if not isinstance(current, dict):\n            return default\n        ",
  "def format_bytes(size:int) -> str:\n    for unit in ['B', 'KB', 'MB', 'GB']:\n        if size < 1024:\n            ",
  "def truncate(text:str, max_len:int, suffix:str='...') -> str:\n    if len(text) <= max_len:\n        return text\n    ",
  "def group_by(items:list, key:str) -> dict:\n    result = {}\n    for item in items:\n        group = item.get(key)\n        if group not in result:\n            result[group] = []\n        ",
  "def moving_average(values:list, window:int) -> list:\n    result = []\n    for i in range(len(values) - window + 1):\n        window_vals = values[i:i + window]\n        ",
  "def unique(items:list) -> list:\n    seen = set()\n    result = []\n    for item in items:\n        if item not in seen:\n            ",
  "def batch(iterable, n:int):\n    batch = []\n    for item in iterable:\n        batch.append(item)\n        if len(batch) == n:\n            yield batch\n            ",
  "def count_words(text:str) -> dict:\n    counts = {}\n    for word in text.lower().split():\n        if word not in counts:\n            counts[word] = 0\n        ",
  "def normalize(values:list) -> list:\n    min_val = min(values)\n    max_val = max(values)\n    if max_val == min_val:\n        ",
  "def find_duplicates(lst:list) -> list:\n    seen = set()\n    duplicates = []\n    for item in lst:\n        if item in seen:\n            ",
  "def sliding_window(lst:list, size:int):\n    for i in range(len(lst) - size + 1):\n        ",
  "def tokenize(text:str, lowercase:bool=True) -> list:\n    if lowercase:\n        text = text.lower()\n    tokens = []\n    current = ''\n    for char in text:\n        if char.isalnum():\n            ",
  "def cosine_similarity(vec1:list, vec2:list) -> float:\n    dot = sum(a * b for a, b in zip(vec1, vec2))\n    norm1 = sum(a ** 2 for a in vec1) ** 0.5\n    norm2 = sum(b ** 2 for b in vec2) ** 0.5\n    if norm1 == 0 or norm2 == 0:\n        ",
  "def softmax(logits:list) -> list:\n    max_val = max(logits)\n    exps = [2.718 ** (x - max_val) for x in logits]\n    total = sum(exps)\n    ",
  "def load_jsonl(file_path:str) -> list:\n    records = []\n    with open(file_path) as f:\n        for line in f:\n            line = line.strip()\n            if line:\n                ",
  "def split_dataset(data:list, train_ratio:float=0.8) -> tuple:\n    split_idx = int(len(data) * train_ratio)\n    ",
  "def compute_accuracy(predictions:list, labels:list) -> float:\n    if len(predictions) != len(labels):\n        raise ValueError('Length mismatch')\n    correct = 0\n    for pred, label in zip(predictions, labels):\n        ",
  "def top_k(scores:list, k:int) -> list:\n    indexed = [(score, i) for i, score in enumerate(scores)]\n    indexed.sort(reverse=True)\n    ",
  "def running_mean(values:list) -> list:\n    result = []\n    total = 0\n    for i, val in enumerate(values):\n        total += val\n        ",
  "def safe_divide(a:float, b:float, default:float=0.0) -> float:\n    if b == 0:\n        ",
  "def levenshtein(s1:str, s2:str) -> int:\n    dp = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]\n    for i in range(len(s1) + 1):\n        dp[i][0] = i\n    for j in range(len(s2) + 1):\n        dp[0][j] = j\n    for i in range(1, len(s1) + 1):\n        for j in range(1, len(s2) + 1):\n            if s1[i-1] == s2[j-1]:\n                ",
  "def rotate(lst:list, k:int) -> list:\n    if not lst:\n        return lst\n    k = k % len(lst)\n    ",
  "def exponential_backoff(attempt:int, base:float=2.0, max_wait:float=60.0) -> float:\n    wait = base ** attempt\n    ",
  "def one_hot_encode(label:int, num_classes:int) -> list:\n    result = [0] * num_classes\n    if 0 <= label < num_classes:\n        ",
  "def confusion_matrix(predictions:list, labels:list, num_classes:int) -> list:\n    matrix = [[0] * num_classes for _ in range(num_classes)]\n    for pred, label in zip(predictions, labels):\n        ",
  "def snake_to_camel(s:str) -> str:\n    parts = s.split('_')\n    ",
  "def camel_to_snake(s:str) -> str:\n    result = []\n    for i, char in enumerate(s):\n        if char.isupper() and i > 0:\n            ",
  "def clamp(value:float, min_val:float, max_val:float) -> float:\n    if value < min_val:\n        ",
  "def zip_to_dict(keys:list, values:list) -> dict:\n    if len(keys) != len(values):\n        raise ValueError('Length mismatch')\n    ",
  "def debounce(func, wait:float):\n    last_called = [0]\n    def wrapper(*args, **kwargs):\n        import time\n        now = time.time()\n        if now - last_called[0] >= wait:\n            ",
  "def is_palindrome(s:str) -> bool:\n    s = s.lower().replace(' ', '')\n    left, right = 0, len(s) - 1\n    while left < right:\n        ",
  "def save_jsonl(records:list, file_path:str) -> None:\n    with open(file_path, 'w') as f:\n        for record in records:\n            ",
  "def parse_config(file_path:str, encoding:str='utf-8'):\n    ",
  "def retry(func, max_attempts:int, delay:float):\n    ",
  "def read_csv(file_path:str):\n    with open(file_path) as f:\n        headers = f.readline().strip().split(',')\n        rows = []\n        for line in f:\n            ",
  "def validate_email(email:str) -> bool:\n    ",
  "def flatten(nested:list) -> list:\n    ",
  "def chunk(lst:list, size:int) -> list:\n    ",
  "def merge_dicts(base:dict, override:dict) -> dict:\n    ",
  "def safe_divide(a:float, b:float, default:float=0.0) -> float:\n    ",
  "def paginate(items:list, page:int, page_size:int) -> list:\n    ",
  "def debounce(func, wait:float):\n    ",
  "def deep_get(obj:dict, *keys, default=None):\n    ",
  "def format_bytes(size:int) -> str:\n    ",
  "def truncate(text:str, max_len:int, suffix:str='...') -> str:\n    ",
  "def is_palindrome(s:str) -> bool:\n    ",
  "def group_by(items:list, key:str) -> dict:\n    ",
  "def moving_average(values:list, window:int) -> list:\n    ",
  "def clamp(value:float, min_val:float, max_val:float) -> float:\n    ",
  "def unique(items:list) -> list:\n    ",
  "def zip_to_dict(keys:list, values:list) -> dict:\n    ",
  "def batch(iterable, n:int):\n    ",
  "def snake_to_camel(s:str) -> str:\n    ",
  "def camel_to_snake(s:str) -> str:\n    ",
  "def count_words(text:str) -> dict:\n    ",
  "def normalize(values:list) -> list:\n    ",
  "def levenshtein(s1:str, s2:str) -> int:\n    ",
  "def find_duplicates(lst:list) -> list:\n    ",
  "def rotate(lst:list, k:int) -> list:\n    ",
  "def sliding_window(lst:list, size:int):\n    ",
  "def exponential_backoff(attempt:int, base:float=2.0, max_wait:float=60.0) -> float:\n    ",
  "def tokenize(text:str, lowercase:bool=True) -> list:\n    ",
  "def cosine_similarity(vec1:list, vec2:list) -> float:\n    ",
  "def softmax(logits:list) -> list:\n    ",
  "def one_hot_encode(label:int, num_classes:int) -> list:\n    ",
  "def load_jsonl(file_path:str) -> list:\n    ",
  "def save_jsonl(records:list, file_path:str) -> None:\n    ",
  "def split_dataset(data:list, train_ratio:float=0.8) -> tuple:\n    ",
  "def compute_accuracy(predictions:list, labels:list) -> float:\n    ",
  "def confusion_matrix(predictions:list, labels:list, num_classes:int) -> list:\n    ",
  "def top_k(scores:list, k:int) -> list:\n    ",
  "def running_mean(values:list) -> list:\n    ",
];

function getHash() {
  return window.location.hash.slice(1) || 'home';
}

function showPage(id) {
  window.location.hash = id;
}

function renderPage() {
  const id = getHash();
  if (id.startsWith('post/')) {
    loadPost(id.replace('post/', ''));
    return;
  }
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  const target = document.getElementById(id);
  if (target) target.classList.add('active');
}

window.addEventListener('hashchange', renderPage);
window.addEventListener('load', renderPage);

function setTransInput(text) {
  document.getElementById('trans-input').value = text;
}

function shuffleCodeExample() {
  const idx = Math.floor(Math.random() * CODE_EXAMPLES.length);
  document.getElementById('code-input').value = CODE_EXAMPLES[idx];
}

async function runTranslation() {
  const sentence = document.getElementById('trans-input').value.trim();
  if (!sentence) return;
  const variant = document.getElementById('trans-variant').value;
  const max_length = parseInt(document.getElementById('trans-maxlen').value);
  const temperature = parseFloat(document.getElementById('trans-temp').value);
  const top_p = parseFloat(document.getElementById('trans-topp').value);
  const out = document.getElementById('trans-output');
  out.innerHTML = '<span class="loading-dot">▌</span>';
  try {
    const res = await fetch(API + '/translate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sentence, variant, max_length, temperature, top_p })
    });
    const data = await res.json();
    out.textContent = data;
  } catch (e) {
    out.innerHTML = '<span style="color:#ef4444">Could not reach the model server.</span>';
  }
}

async function runCoding() {
  const code = document.getElementById('code-input').value.trim();
  if (!code) return;
  const variant = document.getElementById('code-variant').value;
  const max_length = parseInt(document.getElementById('code-maxlen').value);
  const temperature = parseFloat(document.getElementById('code-temp').value);
  const top_p = parseFloat(document.getElementById('code-topp').value);
  const repetition_penalty = parseFloat(document.getElementById('code-reppentalty').value);
  const out = document.getElementById('code-output');
  out.innerHTML = '<span class="loading-dot">▌</span>';
  try {
    const res = await fetch(API + '/coding', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ code, variant, max_length, temperature, top_p, repetition_penalty })
    });
    const data = await res.json();
    out.textContent = data;
  } catch (e) {
    out.innerHTML = '<span style="color:#ef4444">Could not reach the model server.</span>';
  }
}

async function loadPost(slug) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  const postPage = document.getElementById('post-page');
  postPage.classList.add('active');
  postPage.innerHTML = '<p style="color:#94A3B8; font-size:14px;">Loading...</p>';
  try {
    const res = await fetch(API + '/writing/' + slug);
    const data = await res.json();
    postPage.innerHTML = `
      <div class="post-header">
        <a class="back-btn" onclick="showPage('writing')">← writing</a>
        <div class="font-toggle">
          <button onclick="setFont('sans')" id="font-sans-btn" class="font-btn active">Sans</button>
          <button onclick="setFont('serif')" id="font-serif-btn" class="font-btn">Serif</button>
          <button onclick="setFont('mono')" id="font-mono-btn" class="font-btn">Mono</button>
        </div>
      </div>
      <article class="post-body" id="post-body">${data.html}</article>
    `;
    if (window.hljs) hljs.highlightAll();
    if (window.renderMathInElement) {
      renderMathInElement(postPage, {
        delimiters: [
          { left: '$$', right: '$$', display: true },
          { left: '$', right: '$', display: false }
        ]
      });
    }
  } catch (e) {
    postPage.innerHTML = '<p style="color:#ef4444">Could not load post.</p>';
  }
}

function setFont(type) {
  const body = document.getElementById('post-body');
  body.className = 'post-body font-' + type;
  document.querySelectorAll('.font-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('font-' + type + '-btn').classList.add('active');
}