import os
from taipy import Gui
from taipy.gui import notify
from pipeline import process_image

page = """
<|layout|columns=1 1|
<|layout|columns=1 1|
<|{uploaded_img}|file_selector|label= Загрузить изображение|drop_message=Drop here to Upload|extensions=.png,.jpg|>
<|part|render={render_button}|
<|Распознать текст|button|class_name=plain|on_action=on_button_action|>
|>
|>

<|
### 
|>
|>

<|layout|columns=1 1|
<|
<|part|render={render_uploaded}|
<|{uploaded_img}|image|width=74%|>
|>
<|part|render={render_steges}|
<|{hwt_section_img}|image|width=74%|>

<|{nolines_img}|image|width=74%|>

<|{words_selected_img}|image|width=74%|>
|>
|>

<|
<|
###<|{status_text}|>
|>
<|{result_text}|>
|>
|>
"""

stages_folder_path = "stages"
hwt_section_img = os.path.join(stages_folder_path , "selection.jpg")
nolines_img = os.path.join(stages_folder_path , "nolines.jpg")
words_selected_img = os.path.join(stages_folder_path , "segmentation.jpg")

render_uploaded = True
render_steges = False
render_button = False
uploaded_img = ""
recognized_text = ""
result_text = ""
status_text = ""


def on_button_action(state):
    notify(state, 'info', f'Process started')
    state.status_text = "Распознавание запущено"
    result = process_image(state.uploaded_img)
    notify(state, 'success', f'Process completed')
    state.render_steges = True
    state.render_uploaded = False
    state.status_text = "Результат:"
    state.result_text = result


def on_change(state, var, val):
    if var == "uploaded_img":
        state.render_steges = False
        state.render_uploaded = True
        state.render_button = True
        return


Gui(page).run(use_reloader=True, dark_mode=False)
