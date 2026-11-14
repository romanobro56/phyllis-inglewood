"""In-memory semantic form controls. No network, filesystem, or action selection."""
from html import escape


class ApplicationHarness:
    def __init__(self, fields, facts):
        self.fields = fields
        self.facts = facts
        self.values = {}
        self.submitted = False

    def enter(self, field, value):
        """Execute the brain's chosen type/select operation, including mistakes."""
        if self.submitted:
            raise ValueError("application already submitted")
        if field["kind"] == "select" and value not in field["options"]:
            raise ValueError("not a dropdown option")
        self.values[field["key"]] = value
        return self.correct(field)

    def correct(self, field):
        expected = self.facts.get(field["key"])
        return bool(expected) and self.values.get(field["key"]) == expected

    def complete(self):
        return all(self.correct(f) for f in self.fields if f.get("required", True))

    def submit(self):
        if self.submitted or not self.complete():
            return False
        self.submitted = True
        return True

    def html(self):
        """Inspectable fixture reflecting exactly the same semantic form schema."""
        controls = []
        for field in self.fields:
            key = escape(field["key"], quote=True)
            value = self.values.get(field["key"], "")
            if field["kind"] == "select":
                options = '<option value="">Choose…</option>' + ''.join(
                    '<option{}>{}</option>'.format(' selected' if o == value else '', escape(o))
                    for o in field["options"])
                control = f'<select name="{key}" required>{options}</select>'
            else:
                control = f'<input name="{key}" value="{escape(value, quote=True)}" required>'
            controls.append(f'<label>{key}{control}</label><br>')
        return '<!doctype html><meta charset="utf-8"><title>Elton application fixture</title><form>' + ''.join(controls) + '<button>Submit application</button></form>'
