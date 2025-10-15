import streamlit as st 
import numpy as np
import pandas as pd
import time

'Starting a long computation...'



# Display chat messages from history on app rerun
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # React to user input
# if prompt := st.chat_input("What is up?"):
#     # Display user message in chat message container
#     with st.chat_message("user"):
#         st.markdown(prompt)
#     # Add user message to chat history
#     st.session_state.messages.append({"role": "user", "content": prompt})
    
# # Display assistant response in chat message container
# with st.chat_message("assistant"):
#     response = st.write_stream(response_generator())
# # Add assistant response to chat history
# st.session_state.messages.append({"role": "assistant", "content": response})

# with st.chat_message("assistant"):
#     st.write("Hello 👋")
#     st.bar_chart(np.random.randn(30, 3))
    
# text_recived = st.chat_message("assistant")
# # message.write("Hello my assistant 👋")
# # message.bar_chart(np.random.randn(40, 8))

# prompt = st.chat_input("Say something")
# st.session_state.messages.append({"role": "user", "content": prompt})
# if prompt:
#     text_recived.write(f"User has sent the following prompt: {prompt}")
    
    

# Add a placeholder
latest_iteration = st.empty()
bar = st.progress(0)

for i in range(100):
  # Update the progress bar with each iteration.
  latest_iteration.text(f'Iteration {i+1}')
  bar.progress(i + 1)
  time.sleep(0.1)

'...and now we\'re done!'

"""
# My first app
Here's our first attempt at using data to create a table:
"""
import streamlit as st
x = st.slider('x')  # 👈 this is a widget
st.write(x, 'squared is', x * x)

if st.checkbox('Show dataframe'):
    chart_data = pd.DataFrame(
       np.random.randn(20, 3),
       columns=['a', 'b', 'c'])

    chart_data
    

df = pd.DataFrame({
  'first column': [1, 2, 3, 4, 4,4,4,6,7,7,5],
  'second column': [1, 2, 3, 4, 4,4,4,6,7,7,5]
})

st.write(df.head())

"""
## This is static table
"""

st.table(df)

"""
## this is stats table
"""
a = df.describe()
a

"""
Now its 
"""

# dataframe = pd.DataFrame(
#     np.random.randn(10, 20),
#     columns=('col %d' % i for i in range(20)))

# st.dataframe(dataframe.style.highlight_max(axis=0))

dataframe = pd.DataFrame(
    np.random.randn(10, 20),
    columns=('col %d' % i for i in range(20)))
st.table(dataframe)

"""
Now thats a osm map
"""
map_data = pd.DataFrame(
    np.random.randn(500, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])

st.map(map_data)

chart_data = pd.DataFrame(
     np.random.randn(20, 3),
     columns=['a', 'b', 'c'])

st.line_chart(chart_data)