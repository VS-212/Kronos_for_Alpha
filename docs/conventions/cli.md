# CLI Conventions

Стандарт для всех CLI модулей проекта. Новый AI читает этот файл перед созданием или изменением CLI.

## 1. Структура

Каждый CLI модуль = один файл `alpha/<area>/<name>.py` c `if __name__ == "__main__": main()`.

## 2. Флаги (argparse)

Обязательные (`--help` — первый source of truth для AI):

```
Флаг           Тип        Когда обязателен
─────────────────────────────────────────
--start        str        Дата начала (YYYY-MM-DD)
--end          str        Дата конца (YYYY-MM-DD) default: сегодня
--output       Path       Куда писать результат
--dry-run      bool       Показать план, не выполнять
--status       bool       Прочитать результат прошлого запуска
--resume       bool       Продолжить прерванный запуск (default: on)
--config       Path       YAML/JSON конфиг (для сложных модулей)
```

Флаги выставляются в порядке `--input → --control → --output`.

## 3. Артефакты

```
Stdout:         человеку / AI (прогресс, итог, summary)
Файлы:          --output (parquet, npy, json)
Манифест:       --output/manifest.json (машиночитаемый, для --status)
Ошибки:         stderr (пока не разводим, но держим в голове)
```

Манифест (`manifest.json`) — всегда JSON с ключами:
```json
{
  "SBER": {"rows": 43600, "start": "2023-01-03", "end": "2026-04-30"},
  "GAZP": {"rows": 43400, ...}
}
```

## 4. Exit code

```
0 — успех (данные записаны)
1 — ошибка (данные не полные или отсутствуют)
```

`--dry-run` и `--status` всегда возвращают 0.

## 5. Гарантии

Каждый CLI модуль должен реализовать:

```
Resume:     прерванный запуск продолжается, не повторяя готовое
Retry:      временные ошибки (network, rate-limit) — exponential backoff
Idempotent: повторный запуск с теми же флагами даёт тот же результат
Validate:   проверка данных после записи (NaN, дубликаты, границы)
```

## 6. Docstring (known failures)

В начале файла — секция Known failures:

```python
"""Модуль: описание.

Known failures:
  - <точная строка ошибки из лога>
    → <что делать>
"""
```

## 7. Pattern: subagent use

```bash
# Subagent запускает, main agent получает summary
python -m src.data.fetcher --start 2023-01-01 --end 2026-05-01

# Main agent проверяет результат
python -m src.data.fetcher --status
```

## 8. Commit Conventions (AI-friendly)

См. отдельный артефакт: `docs/conventions/commit.md`. Формат: `<type>(M-XXX): description + Contract + Verified`. Каждый коммит = grep-able запись в контрактном логе.

---

*Ported from kronos-alpha/docs/cli-conventions.md. Revision: 1.1.*
