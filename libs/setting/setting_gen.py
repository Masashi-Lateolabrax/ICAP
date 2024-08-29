import os
import warnings

import yaml
import re

_PLACEHOLDER_PATTERN = re.compile(r"\$([\w.]+)")


class _Raw:
    def __init__(self, content: str):
        self.content = content.strip()

    def __str__(self):
        return "<raw:" + self.content + ">"


class _Exec:
    def __init__(self, func_name: str, code: str):
        self.name = func_name.strip()
        self.code = re.sub(r'\n( {2})+', lambda m: "\n" + ' ' * ((len(m.group(0)) - 1) * 2), code.strip())

    def __str__(self):
        return "<exec:" + self.name + ">"


class _Method:
    def __init__(self, name: str, definition: str):
        self.name = name
        self.definition = re.sub(r'\n( {2})+', lambda m: "\n" + ' ' * ((len(m.group(0)) - 1) * 2), definition.strip())

    def __str__(self):
        return "<Method:" + self.name + ">"


def _replace_placeholders(settings, external_funcs, key: str, context: str | list):
    if not isinstance(context, (str | list)):
        return context

    def replace_match(m):
        keys = m.group(1).split(".")
        value = settings
        try:
            for k in keys:
                value = value[k]
            if isinstance(value, _Raw):
                value = value.content
            return str(value).strip()
        except KeyError:
            warnings.warn(f"Not found key; {keys}", stacklevel=2)
            return m.group(0)

    if isinstance(context, list):
        return [_replace_placeholders(settings, external_funcs, key, c) for c in context]
    else:
        if context[0:4] == "raw:":
            return _Raw(_PLACEHOLDER_PATTERN.sub(replace_match, context[4:]))

        elif context[0:5] == "exec:":
            func_name = f"_func{len(external_funcs)}"
            code = context.replace("exec:", f"def {func_name}():")
            exec_ = _Exec(func_name, _PLACEHOLDER_PATTERN.sub(replace_match, code))
            external_funcs.append(exec_)
            return exec_

        elif context[0:5] == "eval:":
            expression = _PLACEHOLDER_PATTERN.sub(replace_match, context[5:])
            return eval(expression)

        elif context[0:7] == "method:":
            code = context.replace("method:", f"def {key}")
            method = _Method(key, _PLACEHOLDER_PATTERN.sub(replace_match, code))
            return method

    return context


def _parser(setting_file_path):
    with open(setting_file_path, "r") as file:
        settings = yaml.safe_load(file)
        external_funcs: list[_Exec] = []

        queue: list[dict] = [settings]
        while len(queue) > 0:
            target = queue.pop(0)
            for key in target.keys():
                value = target[key]
                if isinstance(value, dict):
                    queue.append(value)
                else:
                    target[key] = _replace_placeholders(settings, external_funcs, key, value)

        return settings, external_funcs


def _save(settings: dict, external_funcs: list[_Exec], output_file_path: str):
    with open(output_file_path, "w") as file:
        import_ = settings.pop("Import", dict())
        if len(import_) > 0:
            for k, v in import_.items():
                if k == v:
                    file.write(f"import {k}")
                else:
                    file.write(f"import {k} as {v}")
                file.write("\n")
            file.write("\n")

        for f in external_funcs:
            file.writelines(f.code)
            file.write("\n\n")

        def write_settings(k, v, o):
            if k is None:
                file.write(o + str(v))

            elif isinstance(v, dict):
                file.write(f"{o}class {k}:\n")
                for kk, vv in v.items():
                    write_settings(kk, vv, o + "    ")

            elif isinstance(v, (int, float)):
                file.write(f"{o}{k} = {v}\n")

            elif isinstance(v, str):
                file.write(f"{o}{k} = \"{v}\"\n")

            elif isinstance(v, list):
                file.write(f"{o}{k} = [")
                for vv in v[0:-1]:
                    write_settings(None, vv, "")
                    file.write(", ")
                write_settings(None, v[-1], "")
                file.write("]\n")

            elif isinstance(v, _Raw):
                file.write(f"{o}{k} = {v.content}\n")

            elif isinstance(v, _Exec):
                file.write(f"{o}{k} = {v.name}()\n")

            elif isinstance(v, _Method):
                file.write(f"{o}@staticmethod\n")
                for l in v.definition.split("\n"):
                    file.write(f"{o}{l}\n")

            else:
                warnings.warn("Unsupported type.")

        write_settings("Settings", settings, "")


def generate(setting_file_path: str, out: str = None):
    if out is None:
        d = os.path.dirname(
            os.path.abspath(setting_file_path)
        )
        out = os.path.join(d, "settings.py")

    settings, external_funcs = _parser(setting_file_path)
    _save(settings, external_funcs, out)


def _test():
    file_path = "./settings.yaml"

    settings, external_funcs = _parser(file_path)

    for f in external_funcs:
        print(f.code)
    print()

    queue = [("", settings)]
    while len(queue) > 0:
        offset, target = queue.pop(0)
        items = list(target.items()) if isinstance(target, dict) else target
        for i, (k, v) in enumerate(items):
            if isinstance(v, dict):
                print(f"{k}")
                queue.insert(0, (offset, items[i + 1:]))
                queue.insert(0, (offset + " ", v))
                break
            elif isinstance(v, (str, int, float, list, _Raw, _Exec)):
                print(f"{offset}{k} = {v}")

    _save(settings, external_funcs, file_path)


if __name__ == '__main__':
    _test()
