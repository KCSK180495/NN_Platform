"""Check package wiring without importing the GUI or training runtime."""

import ast
import builtins
import symtable
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "nn_training_studio"


class ArchitectureTests(unittest.TestCase):
    def test_module_globals_are_resolvable(self):
        for path in PACKAGE.rglob("*.py"):
            table = symtable.symtable(path.read_text(), str(path), "exec")
            available = set(table.get_identifiers()) | set(vars(builtins)) | {
                "__file__", "__name__", "__package__", "__doc__", "__spec__",
            }
            def check(scope):
                for symbol in scope.get_symbols():
                    if symbol.is_global() and symbol.is_referenced():
                        self.assertIn(symbol.get_name(), available,
                                      f"{path.name}: missing {symbol.get_name()}")
                for child in scope.get_children():
                    check(child)
            with self.subTest(module=path.name):
                check(table)

    def test_internal_imports_exist_and_have_no_cycles(self):
        modules = {
            ".".join(path.relative_to(ROOT).with_suffix("").parts): path
            for path in PACKAGE.rglob("*.py")
        }
        trees = {name: ast.parse(path.read_text()) for name, path in modules.items()}
        exports = {}
        for name, path in modules.items():
            table = symtable.symtable(path.read_text(), str(path), "exec")
            exports[name] = set(table.get_identifiers())
        edges = {name: set() for name in modules}
        for name, tree in trees.items():
            for node in tree.body:
                if not isinstance(node, ast.ImportFrom):
                    continue
                if node.level:
                    package = name.rsplit(".", node.level)[0]
                    target = package + "." + (node.module or "")
                else:
                    target = node.module or ""
                if not target.startswith("nn_training_studio."):
                    continue
                self.assertIn(target, modules)
                edges[name].add(target)
                for alias in node.names:
                    self.assertIn(alias.name, exports[target], f"{name}: {alias.name}")
        done = set()
        def visit(name, stack):
            self.assertNotIn(name, stack, f"Circular import: {stack} -> {name}")
            if name in done:
                return
            for target in edges[name]:
                visit(target, stack + [name])
            done.add(name)
        for name in modules:
            visit(name, [])

    def test_wizard_mixins_do_not_shadow_each_other(self):
        seen = set()
        for path in sorted((PACKAGE / "ui").glob("wizard_*.py")):
            tree = ast.parse(path.read_text())
            cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
            for node in cls.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    self.assertNotIn(node.name, seen, f"Duplicate method in {path.name}")
                    seen.add(node.name)


if __name__ == "__main__":
    unittest.main()
