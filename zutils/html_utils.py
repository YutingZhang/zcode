

__all__ = [
    'replace_token_with_indent',
]

def replace_token_with_indent(template: str, token: str, content: str):
    indent_token = '<%%INDENT%%>'
    c = indent_token + f"\n{indent_token}".join(content.split('\n'))
    all_content = template.replace(token, c)
    tl = all_content.split('\n')
    indent = ''
    for i, t in enumerate(tl):
        if t.startswith(indent_token):
            tl[i] = t.replace(indent_token, indent)
        else:
            st = t.lstrip()
            if indent_token in st:
                indent = t[:-len(st)]
                tl[i] = t.replace(indent_token, '')
    return "\n".join(tl)
