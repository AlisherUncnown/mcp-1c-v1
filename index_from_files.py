"""
Генератор RAG-индекса из файловой выгрузки конфигурации 1С.
Создаёт ZIP с objects.csv + markdown файлами для загрузки через upload_zip.py.

Использование:
  python index_from_files.py <путь_к_выгрузке> <имя_коллекции>

Пример:
  python index_from_files.py "C:/Users/user/Documents/Садыхан" sadykhan_roznica
"""

import os
import sys
import csv
import io
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

# Папка выгрузки -> тип объекта в единственном числе, как в 1С и в фильтре MCP-сервера
TYPE_MAP = {
    "CommonModules":               "ОбщийМодуль",
    "DataProcessors":              "Обработка",
    "Catalogs":                    "Справочник",
    "Documents":                   "Документ",
    "InformationRegisters":        "РегистрСведений",
    "AccumulationRegisters":       "РегистрНакопления",
    "AccountingRegisters":         "РегистрБухгалтерии",
    "CalculationRegisters":        "РегистрРасчета",
    "ExchangePlans":               "ПланОбмена",
    "EventSubscriptions":          "ПодпискаНаСобытие",
    "CommonForms":                 "ОбщаяФорма",
    "Reports":                     "Отчет",
    "BusinessProcesses":           "БизнесПроцесс",
    "Tasks":                       "Задача",
    "ChartsOfAccounts":            "ПланСчетов",
    "ChartsOfCharacteristicTypes": "ПланВидовХарактеристик",
    "ChartsOfCalculationTypes":    "ПланВидовРасчета",
    "Constants":                   "Константа",
    "Enums":                       "Перечисление",
    "DocumentJournals":            "ЖурналДокументов",
    "Sequences":                   "Последовательность",
    "ScheduledJobs":               "РегламентноеЗадание",
    "SessionParameters":           "ПараметрСеанса",
    "Roles":                       "Роль",
    "CommonAttributes":            "ОбщийРеквизит",
    "Subsystems":                  "Подсистема",
    "FilterCriteria":              "КритерийОтбора",
    "FunctionalOptions":           "ФункциональнаяОпция",
    "HTTPServices":                "HTTPСервис",
    "WebServices":                 "WebСервис",
    "XDTOPackages":                "ПакетXDTO",
    "DefinedTypes":                "ОпределяемыйТип",
    "CommonCommands":              "ОбщаяКоманда",
}

MD = "{http://v8.1c.ru/8.3/MDClasses}"
V8 = "{http://v8.1c.ru/8.1/data/core}"

# Дочерние элементы метаданных, которые попадают в описание объекта
CHILD_KINDS = {
    "Attribute":       "Реквизит",
    "TabularSection":  "Табличная часть",
    "Dimension":       "Измерение",
    "Resource":        "Ресурс",
    "EnumValue":       "Значение",
    "AccountingFlag":  "Признак учета",
    "Command":         "Команда",
}

CODE_LIMIT = 8000  # символов на модуль, чтобы не раздувать payload Qdrant


def synonym_of(props):
    """Русский синоним из узла <Properties> (с учётом пространства имён MDClasses)."""
    if props is None:
        return ""
    syn = props.find(f"{MD}Synonym")
    if syn is None:
        return ""
    fallback = ""
    for item in syn.findall(f"{V8}item"):
        lang = item.findtext(f"{V8}lang") or ""
        content = (item.findtext(f"{V8}content") or "").strip()
        if content and lang == "ru":
            return content
        if content and not fallback:
            fallback = content
    return fallback


def read_metadata(xml_path):
    """Имя, синоним и дочерние элементы (реквизиты, ТЧ, измерения...) объекта."""
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return None, "", []
    obj = next(iter(root), None)  # <Document>, <Catalog>, ...
    if obj is None:
        return None, "", []
    props = obj.find(f"{MD}Properties")
    name = props.findtext(f"{MD}Name") if props is not None else None
    children = []
    child_objects = obj.find(f"{MD}ChildObjects")
    if child_objects is not None:
        for ch in child_objects:
            kind = CHILD_KINDS.get(ch.tag.replace(MD, ""))
            if not kind:
                continue
            cprops = ch.find(f"{MD}Properties")
            cname = cprops.findtext(f"{MD}Name") if cprops is not None else ch.text
            if not cname:
                continue
            children.append((kind, cname.strip(), synonym_of(cprops)))
            if kind == "Табличная часть":
                ts_children = ch.find(f"{MD}ChildObjects")
                for a in (ts_children if ts_children is not None else []):
                    aprops = a.find(f"{MD}Properties")
                    aname = aprops.findtext(f"{MD}Name") if aprops is not None else None
                    if aname:
                        children.append((f"  реквизит ТЧ {cname}", aname.strip(), synonym_of(aprops)))
    return name, synonym_of(props), children


def module_context(bsl_path):
    p = bsl_path.replace("\\", "/")
    low = p.lower()
    if "objectmodule" in low:
        return "Модуль объекта"
    if "managermodule" in low:
        return "Модуль менеджера"
    if "recordsetmodule" in low:
        return "Модуль набора записей"
    if "/forms/" in low:
        return "Форма: " + p.split("/Forms/")[1].split("/")[0]
    if "/commands/" in low:
        return "Команда: " + p.split("/Commands/")[1].split("/")[0]
    return "Модуль"


def build_doc(obj_type, obj_name, synonym, children, bsl_paths, xml_text=None):
    parts = [f"# {obj_type}: {obj_name}"]
    if synonym:
        parts.append(f"**Синоним:** {synonym}\n")
    if children:
        parts.append("## Состав\n")
        for kind, cname, csyn in children:
            parts.append(f"- {kind}: {cname}" + (f" ({csyn})" if csyn else ""))
    if xml_text is not None:
        parts.append(f"\n```xml\n{xml_text[:3000]}\n```")
    for bsl_path in bsl_paths:
        try:
            with open(bsl_path, "r", encoding="utf-8-sig") as f:
                code = f.read().strip()
        except Exception as e:
            parts.append(f"\n## {module_context(bsl_path)}\n\n*Ошибка чтения: {e}*")
            continue
        if code:
            if len(code) > CODE_LIMIT:
                code = code[:CODE_LIMIT] + "\n... (truncated)"
            parts.append(f"\n## {module_context(bsl_path)}\n\n```bsl\n{code}\n```")
    return "\n".join(parts)


def collect_objects(config_path):
    """[(obj_name, obj_type, synonym, md_filename, md_content)] по всем типам метаданных."""
    config_path = Path(config_path)
    objects = []
    for folder_name, type_name in TYPE_MAP.items():
        type_dir = config_path / folder_name
        if not type_dir.exists():
            continue
        # Объект = XML-файл описания; код (если есть) лежит в одноимённой папке
        for xml_file in sorted(type_dir.glob("*.xml")):
            name, synonym, children = read_metadata(xml_file)
            obj_name = name or xml_file.stem
            obj_dir = type_dir / xml_file.stem
            bsl_files = sorted(str(p) for p in obj_dir.rglob("*.bsl")) if obj_dir.is_dir() else []
            xml_text = None
            if folder_name == "EventSubscriptions":
                xml_text = xml_file.read_text(encoding="utf-8-sig", errors="replace")
            md = build_doc(type_name, obj_name, synonym, children, bsl_files, xml_text)
            objects.append((obj_name, type_name, synonym, f"{folder_name}_{obj_name}.md", md))
    return objects


def index_config(config_path, collection_name, output_zip=None):
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"ОШИБКА: путь не существует: {config_path}")
        sys.exit(1)
    if output_zip is None:
        output_zip = f"{collection_name}.zip"

    print(f"\nИндексация: {config_path}")
    print(f"Коллекция:  {collection_name}")
    objects = collect_objects(config_path)
    with_syn = sum(1 for o in objects if o[2])
    print(f"Найдено объектов: {len(objects)}, с синонимом: {with_syn}")
    if not objects:
        print("ОШИБКА: объекты не найдены. Проверьте путь к выгрузке.")
        sys.exit(1)

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=["Имя объекта", "Тип объекта", "Синоним", "Файл"],
                                delimiter=";", quotechar='"', quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for obj_name, obj_type, synonym, md_filename, _ in objects:
            writer.writerow({"Имя объекта": obj_name, "Тип объекта": obj_type,
                             "Синоним": synonym, "Файл": md_filename})
        zf.writestr("objects.csv", buf.getvalue())
        for _, _, _, md_filename, md_content in objects:
            zf.writestr(md_filename, md_content)

    size_mb = os.path.getsize(output_zip) / 1024 / 1024
    print(f"ZIP: {output_zip} ({size_mb:.1f} MB)")
    print(f"Загрузка в Qdrant: python upload_zip.py \"{output_zip}\" {collection_name}")
    return output_zip


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Использование: python index_from_files.py <путь_к_выгрузке> <имя_коллекции>")
        sys.exit(0)
    index_config(sys.argv[1], sys.argv[2])
