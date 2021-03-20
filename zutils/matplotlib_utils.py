import os
from z_python_utils.io import mkpdir_p


def preprocess_and_get_figure_filename(fn: str) -> str:
    output_path = fn
    if fn.endswith(".html"):
        rel_output_path = os.path.join(
            "_figures", os.path.splitext(os.path.basename(fn))[0] + ".jpg"
        )
        output_path = os.path.join(os.path.dirname(fn), rel_output_path)
        with open(fn, "w") as f:
            print("""
    <html>
    <header>
        <style>
            html {padding: 0; margin: 0}
            body {padding: 0; margin: 0; overflow-x: hidden;}
            .main-picture {padding: 0; margin: 0 1%; width: 98%;}
        </style>
    </header>
    <body>
    <img class="main-picture" src="$$$IMAGE_PATH$$$">
    </body>
    </html>
    """.replace("$$$IMAGE_PATH$$$", rel_output_path),
                  file=f)

    mkpdir_p(output_path)

    return output_path
