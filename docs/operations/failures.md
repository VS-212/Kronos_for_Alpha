# Modal: Known Failures

Каталог отказов для Modal-джобов (`train_job.py`, `backtest_job.py`).

**Правило**: если AI встречает новую ошибку — добавить запись в этот файл с exact symptom из лога.

---

## ModalTimeoutError: Function runs longer than configured timeout

Симптом:
```
modal.exception.ModalTimeoutError: Function runs longer than configured timeout
```

Причина: Modal job превысил лимит времени.
- Fine-tune 30 эпох на A100 ≈ 2-4h
- Modal default timeout = 1800s (30 min)

Фикс:
1. В `train_job.py` добавить: `@app.function(timeout=28800)` (8 часов)
2. В `modal run`: `--timeout 28800`
3. Сhar checkpoint каждые 100 steps — `kronos_moex_latest.pt` (heartbeat)
4. Resume через `--resume kronos_moex_best.pt`

---

## FileNotFoundError: checkpoint directory

Симптом:
```
FileNotFoundError: [Errno 2] No such file or directory: '/checkpoints/kronos_moex/'
```

Причина: volume не примонтирован или путь не существует в образе.

Фикс:
1. Проверить `image.py`:
```python
volume = modal.Volume.from_name("kronos-checkpoints", create_if_missing=True)
@app.function(volumes={"/checkpoints": volume})
```
2. Убедиться что директория создаётся при старте:
```python
os.makedirs("/checkpoints/kronos_moex", exist_ok=True)
```
3. Если volume stale: `modal volume reload kronos-checkpoints`

---

## CUDA Out of Memory

Симптом:
```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.1 GiB
```

Причина: batch size слишком большой для A100 40GB.

Фикс:
1. Уменьшить batch: `batch=8` или `batch=4`
2. Уменьшить L: `L=256` вместо `512`
3. Включить gradient checkpointing
4. Проверить что не загружены лишние тензоры (особенно attention KV cache)

---

## Torch not compiled with CUDA

Симптом:
```
AssertionError: Torch not compiled with CUDA enabled
```

Причина: PyTorch установлен CPU-only (неправильный wheel).

Фикс:
1. В `image.py` указать явно: `pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124`
2. Проверить `pip list | grep torch` в логах сборки

---

## App not found in file

Симптом:
```
Error: No @app found in alpha/infra/train_job.py
```

Причина: Modal не может найти `app = modal.App()` в файле.

Фикс:
1. Убедиться что в `train_job.py` есть глобальная переменная:
```python
app = modal.App("kronos-train")
```
2. Функции декорированы `@app.function()`, не `@modal.app.function()`

---

## pip install timeout (image build)

Симптом:
```
WARNING: Retrying (Retry(total=4, ...)) after connection broken
ReadTimeoutError: HTTPSConnectionPool: Read timed out.
```

Причина: загрузка torch (2-4GB) через pip падает по таймауту.

Фикс:
1. Увеличить pip default timeout: `.pip_install("torch==2.4.0", env={"PIP_DEFAULT_TIMEOUT": "300"})`  
2. Если ошибка повторяется — `modal image rebuild --force` (сбросить кэш)

---

## Stale image — изменения не применяются

Симптом:
```
# Код изменили, установили новую библиотеку, но при запуске — ModuleNotFoundError
ModuleNotFoundError: No module named 'xxx'
```

Причина: Modal кэширует image. Если hash изменился незначительно (комментарий, пробел), Modal не пересобирает.

Фикс:
1. `modal image rebuild` — принудительная пересборка
2. Или изменить содержимое `pip_install` (добавить комментарий внутрь строки)
3. Проверить что изменения действительно в image: `modal image ls`

---

## ModuleNotFoundError: modal

Симптом:
```
ModuleNotFoundError: No module named 'modal'
```

Причина: `modal` пакет не установлен локально (не в образе, а на машине, откуда запускается `modal run`).

Фикс:
```bash
pip install modal
modal setup  # только первый раз
```

---

## HuggingFace token required

Симптом:
```
OSError: Can't load tokenizer for 'NeoQuasar/KronosTokenizer-2k'.
If you were trying to load it from 'https://huggingface.co/models', make sure you
don't have a local directory with the same name. Otherwise, make sure
'NeoQuasar/KronosTokenizer-2k' is the correct path to a directory containing a config.json
```

Причина: модель на HuggingFace требует авторизации (gated model).

Фикс:
1. Получить token: https://huggingface.co/settings/tokens
2. Передать в Modal через Secret:
```python
import modal
hf_secret = modal.Secret.from_name("huggingface")
# → в modal run: modal secret create huggingface HF_TOKEN=hf_xxx
```

---

## Concurrent checkpoint writes

Симптом:
```
# Два Modal jobs запущены одновременно
# Чекпоинт перезаписан вторым job'ом, loss скачет
```

Причина: нет блокировки на checkpoint.

Фикс:
1. Каждый job пишет уникальный: `kronos_moex_epoch_{epoch}.pt`
2. Общий `kronos_moex_best.pt` только через atomic flag
3. Не запускать параллельные train-запуски на один volume

---

## Ephemeral disk full

Симптом:
```
OSError: [Errno 28] No space left on device
```

Причина: Modal ephemeral storage ~10GB. Загрузка данных + чекпоинты превысили лимит.

Фикс:
1. Данные хранить на Modal Volume, не на ephemeral
2. Чекпоинты — только на volume
3. Удалять временные файлы после обработки
4. Если нужно больше: `@app.function(cloud="gcp", ...)` или запросить увеличение квоты

---

## Cloud watch — как обнаружить проблему

При падении Modal job:
1. `modal logs kronos-train` — последние логи
2. `modal app list` — статус всех jobs
3. `modal volume ls kronos-checkpoints` — проверить чекпоинты
4. Python `traceback.format_exc()` — exact symptom для этого файла

---

*Ported from kronos-alpha/docs/modal-failures.md. Revision: 1.1.*
