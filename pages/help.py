import streamlit as st
  
st.set_page_config(layout="wide", page_title="Diatribe Help", page_icon="🎧")

st.title(":sparkles: Diatribe Help")

with st.sidebar:
    # table of contents for the help page
    st.markdown("""
        1. [Quickstart](#quickstart)
        1. [Exporting & Importing](#exporting-importing)
        1. [Editing Audio](#editing-audio)
        1. [Dialogue Generation](#dialogue-generation)
        1. [Sidebar](#sidebar)
    """)

st.markdown("""
    ### Diatribe is a tool to create interesting audio dialogues.
""")

st.markdown("""
    The motivation for creating Diatribe was to create a tool for my family. We were having so much
    fun with our cloned voices. At first, I created everything in ElevenLabs and combined and edited
    the dialogue lines in a desktop audio editor. But as family demand for more dialogues continued to grow and
    being the lazy programmer that I am, I decided to create a tool to automate the process. The name `Diatribe`
    was given since the dialogues created tended to be silly and long-winded.
""")

st.info(":bulb: It is worth noting that creating a tool like Diatribe using `Streamlit` has been interesting. While the simplicity of `Streamlit` makes it easy to get started, it can be challenging to build a larger application. Some of the quirks of Diatribe are due to the limitations of `Streamlit`. One thing is for sure, the app definitely feels like a `Streamlit` app.")

# QUICKSTART
st.markdown("# Quickstart")
st.markdown("""
    Getting started with Diatribe is pretty easy. There are a total of four sections to the tool, with two that you fill out as the user and two that are generated for you. The four sections are:
    1. `Characters`: _The speakers of your dialogue_
    1. `Dialogue`: _The text of your dialogue_
    1. `Audio Dialogue`: _The audio of your dialogue_
    1. `Audio Diatribe`: _The dialogue combined into a single audio file_
    
    ### Characters
    The `Characters` section is where you define the speakers of your dialogue. It's also where you decide what voice each character should have. The voices available to you are whatever are provided by ElevenLabs and any cloned voices you may have created. The required fields are `Name` and `Voice`. There are a few optional fields: `Group` and `Description`. These optional fields help with some more advanced features such as dialogue generation using OpenAI and editing the final `Audio Diatribe`.
""")
st.warning(""":warning: Adding characters, removing characters, and modifying character names will clear or reset the dialogue. That being said, you can freely change the voices, groups, and description without affecting the dialogue. Also, changing the sound engine will cause a reset since the voices will be different.
""")
st.image("./images/characters.png", use_column_width=True, caption="Example Characters Section")

st.markdown("""
   ### Dialogue
   The `Dialogue` section is probably the most important section of the tool. The dialogue is what your characters say and eventually determines what audio you will generate. It is also the simplest section to enter, since it consists of only two parts: `Speaker` and `Text`. In other words, who says what and when.
""")
st.image("./images/dialogue.png", use_column_width=True, caption="Example Dialogue Section")

st.markdown("""
    ### Audio Dialogue
    Once you have entered your dialogue, then you can click on the `Generate Audio Dialogue` button. This will use ElevenLabs to convert every line of dialogue into separate audio dialogue lines. This is the section where you can listen to each line of dialogue, and if you are not happy with the result, you can simply click the `Redo` button to have ElevenLabs generate the audio for the dialogue line again. This is really one of the *power features* of Diatribe, since you can keep generating audio dialogue until you are happy with the result of every line.   
""")
st.image("./images/audio_dialogue.png", use_column_width=True, caption="Example Audio Dialogue Section")
st.markdown("""
    It is also important to understand that the audio dialogue supports an iterative process. There are four common changes you will find yourself making during the `Audio Dialogue` section:
    1. Recreating the audio dialogue for a single line of dialogue using the `Redo` button.
    1. Changing the voice of one of your characters and clicking `Generate Audio Dialogue` again.
    1. Changing the text of dialogue lines, which simply requires you to click `Redo` on the lines you have changed.
    1. Adding new dialogue lines, which requires you to click the `Redo` button on the new line (see image below).          
""")
st.image("./images/new_audio_line.png", use_column_width=True, caption="Example New Dialogue Line")
st.warning(":warning: If you delete dialogue lines, you will have to regenerate the audio by clicking the `Generate Audio Dialogue` button.")

st.markdown("""
    ### Audio Diatribe
    Once you are happy with your audio dialogue lines, you can click the `Join Dialogue` button. This will combine all of your audio dialogue lines into a single audio file. This is the section where you can listen to the final audio file and make any changes you want. Once you are happy with the final audio, you can download the final audio file by clicking the `Download Audio Dialogue` button.         
""")
st.markdown("If you are not happy with your final diatribe, simply go back and redo whatever you want. Then click the `Join Dialogue` button again to generate a new final audio file.")
st.image("./images/audio_diatribe.png", use_column_width=True, caption="Example Audio Diatribe Section")

st.markdown("""
     ### :boom: You have now created your first audio diatribe! 
     I hope you have as much fun with Diatribe as my family has had.        
""")

# SAVING & LOADING
st.markdown("# Exporting & Importing")
st.markdown("""
    Diatribe supports exporting and importing projects. This is useful if you want to save your progress and come back to it later. It is also useful if you want to share your project with someone else. The exporting and importing is done using a zip file, which contains all of the information about your project. This includes the dialogue, audio dialogue, and audio diatribe. This is useful if you want to edit the audio files yourself.
""")
st.warning(":warning: Streamlit has a concept of a session. This means when you open a Diatribe in a new tab or reopen your browser, you will have a new session. Each session is unique and does not share your dialogue information. That is a big reason why exporting and importing your project is useful and important. Don't make the mistake of losing your hard work.")
st.markdown("""
    You can find the export and import options at the top of the Diatribe tool. To export your project, click the `Prepare Download` button, then click the `Download` button.
""")
st.image("./images/export_project.png", use_column_width=True, caption="Example Export")
st.markdown("""
    To import a project, click the `Import` tab, then click the `Browse Files`. This will open a file dialog where you can select the zip file you want to import. Once you have selected the zip file, click `Import` and Diatribe will import the project and you can continue working on it.            
""")
st.image("./images/import_project.png", use_column_width=True, caption="Example Import")
st.info("💡 The tool has some sample projects for you to load if you just want to explore.")
st.markdown("""
    If you don't want to save the audio files, you can also just export the characters and dialogue as a text file. This is useful if you want to edit or share just your dialogue text. I won't include a screenshot of this, but you can find the `Export Characters & Dialogue` section below the `Dialogue` section. The main reason you may want to use this option is if you are just getting started with your dialogue project, and you want to manually edit the lines outside of the tool. It's also a small file, so it's easier to work with and share with others.
""")

# EDITING AUDIO
st.markdown("# Editing Audio")
st.markdown("""
    An advanced feature of Diatribe is the ability to edit audio directly in the tool. The tool allows you to edit the audio dialogue lines and the final diatribe. In order to use audio editing, you must click `Enable Audio Editing` in the sidebar under `Dialogue Options`.         
""")
st.image("./images/enable_audio_editing.png", caption="Enable Audio Editing")
st.markdown("""
    By clicking on `Edit Audio` below an audio dialogue line, you can edit the audio for that line. This will open a new section. The edit audio section includes `Basic`, `Soundboard`, and `Special Effect` tabs.            
""")
st.image("./images/edit_audio_line.png", use_column_width=True, caption="Dialogue Line Edit Audio")
st.markdown("""
    There are so many options under this section that I won't go into detail about all of them. I will just say that you can do a lot of cool things with the audio editing. You can perform basic audio edits like adjusting the volume or trimming the audio. You can also add cool sounds by adding chorus, distortion, reverb. or a special effect like an explosion in the background.
""")

st.markdown("""
    You can also apply audio edits to the final diatribe. This is useful if you want to add a background soundtrack or apply normalization to your final audio. To edit the final diatribe, click the `Edit Audio` button under the `Audio Diatribe` section. This will open a similar audio editing section as the audio dialogue lines. The only difference is that the audio editing is focused on audio mastering. The mastering options are `Audiobook Mastering`, `Soundtrack`, and `Soundboard`. Common mastering edits include adding a background soundtrack, adding reverb to groups of characters (makes them sound like they are in the same space), and applying normalization using a `Limiter` and `Compressor`.            
""")
st.image("./images/edit_audio_diatribe.png", use_column_width=True, caption="Audio Diatribe Edit Audio")

st.markdown("# Dialogue Generation")
st.markdown("""
    Sometimes you may need a little help to get started on a dialogue, or you may be stuck and you are not sure how to continue. Diatribe supports dialogue generation using OpenAI. This is an advanced feature that requires an OpenAI API key. If you don't have an OpenAI API key, you can still use the tool, but you will not be able to use the dialogue generation feature. If you do have an OpenAI API key, you can enter it in the `OpenAI Options` section of the sidebar. Once you provide your OpenAI API key, you will see a new `Dialogue Generation` under the `Dialogue` section.          
""")
st.image("./images/dialogue_generation.png", caption="Dialogue Generation")
st.markdown("""
    You will notice in the image above that you can generate a plot and generate the dialogue. This begs the question, how exacty does OpenAI know what to generate? When you ask OpenAI to generate a plot, Diatribe will send OpenAI the character names and their descriptions. This is the reason why `Description` is a column for your characters. When you ask OpenAI to generate dialogue, Diatribe will send OpenAI the character names, their descriptions, and the plot. It is also important to note that OpenAI will not always return the desired number of lines, but it should be close.   
""")
st.info("💡 You can adjust the temperature settings in the sidebar to control the creativity. Also, if you really want a certian number of lines, put the number of lines directly in the plot.")
st.markdown("""
    There is one more use-case where you can use OpenAI: when you have an existing dialogue and you want OpenAI to add new lines. Below the `Dialogue` section, you will find a `Continue Dialogue` button. If you click that, Diatribe will send the character names, their descriptions, the plot, and the existing dialogue to OpenAI. OpenAI will then generate new dialogue lines and add them to the existing dialogue. This is a great way to get unstuck when you are not sure how to continue your dialogue. Keep in mind, the number of lines you have selected in the `Generate Dialogue` section will also be used for the new, continued dialogue lines.            
""")
st.image("./images/continue_dialogue.png", use_column_width=True, caption="Continue Dialogue")

st.markdown("# Sidebar")
st.markdown("The sidebar for Diatribe has a few sections: `Dialogue Options`, `OpenAI Options`, `Voice Explorer`, and `Usage`.")
st.markdown("### Dialogue Options")
st.markdown("The `Dialogue Options` section is where you can enable or disable a few dialogue workflow features: `Enable Instructions` and `Enable Audio Editing`. These options are pretty straight-forward, and once you get more familar with Diatribe, you will usually turn off instructions and turn on audio editing. The rest of the options are configurations for ElevenLabs. I won't go into much detail here, but if you use ElevenLabs, you know what these options are. The most commonly adjusted option is `Stability`, where the lower the number, the more unique sound you will get.")
st.info(":bulb: Each time you click `Generate Audio Dialogue` or `Redo` the current settings will be used. You can use this by tweaking the settings and clicking `Redo` to see how the audio changes.")
st.image("./images/dialogue_options.png", caption="Dialogue Options")

st.markdown("### OpenAI Options")
st.markdown("Most of the OpenAI options are also pretty straight-forward. An important think to remember if that max tokens can be very important when generating dialogue. If you are getting errors when generating a dialogue, it is likely due to the max token being too low, which means you need to switch to a different `Model` and/or increase the `Max Tokens`.")
st.image("./images/openai_options.png", caption="OpenAI Options")

st.markdown("### Exploring Voices")
st.markdown("Clearly, the voices of your characters are important when creating an audio diatribe. Therefore, the `Voice Explorer` was created to make it easy for you to find the perfect voice. There are a number of filtering options you can use. The tool uses the standard voice tags: `accent`, `age`, and `gender`. These are included with all the provided voices by ElevenLabs, and it would be useful if you did the same with your cloned voices. If you do, then the filtering will work with all your voices.")
st.image("./images/voice_explorer.png", caption="Voice Explorer")

st.markdown("### Usage")
st.markdown("The `Usage` section is just a few stats about your ElevenLabs subscription. It's not really that important, but it's nice to have. My family and I were burning through my quota, so I had to upgrade my plan.")
st.image("./images/usage.png", caption="Usage")