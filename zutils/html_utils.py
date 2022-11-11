

__all__ = [
    'replace_token_with_indent',
]

def replace_token_with_indent(template: str, token: str, content: str):
    indent_token = '<%%INDENT%%>'
    c = f"\n{indent_token}".join(content.split('\n'))
    tl = c.split('\n')
    indent = ''
    for i, t in enumerate(tl):
        if t.startswith(indent_token):
            tl[i] = t.replace(indent_token, indent)
        else:
            st = t.lstrip()
            if st:
                indent = t[:-len(st)]
    return "\n".join(tl)
