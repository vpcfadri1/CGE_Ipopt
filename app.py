import streamlit as st
from Ipopt_runner import runner



st.title("Using the CGE Model to Quantify the Impact of Typhoons in the Philippine Economy")
st.write("By Vincent Paul C. Fadri and Herminio L. Gregorio IV")

st.write("")

return_period = st.selectbox("Select the Typhoon Return Period", 
                             options=["5-Year Return Period", 
                                      "10-Year Return Period", 
                                      "20-Year Return Period", 
                                      "50-Year Return Period"])

affected_regions = st.multiselect(
    "Select Affected Regions",
    ["NCR", "CAR", "I", "II", "III",
     "IVA","IVB", "V", "VI", "VII", 
     "VIII", "IX", "X", "XI", "XII", 
    "XIII", "BARMM"]
)

if st.button("Run Model"):
    if not affected_regions:
        st.error("Please select at least one affected region.")
        st.stop()

    with st.spinner("Running the model..."):
        try:
            # After the selectbox
            return_period_value = int(return_period.split("-")[0])
            app_df_Z, app_df_pq = runner(return_period_value, affected_regions)
            st.success("Model run completed successfully!")

            # Display results for Z
            st.subheader(f"Model Result for Gross Domestic Output Z: {return_period}")
            st.dataframe(app_df_Z, use_container_width=True)

            st.subheader(f"Gross Domestic Output Z by Sector: {return_period}")
            st.bar_chart(app_df_Z, 
                         y=["Initial Z", "Shocked Z"], 
                         x_label="Sectors",
                         y_label="Gross Domestic Output in Million Pesos",
                         height=400,
                         color=["#1f77b4", "#ff7f0e"],
                         stack=False)

            # Display results for Price Index (pq)
            st.subheader(f"Model Result for Price of Domestically Consumed Final Goods pq: {return_period}")
            st.dataframe(app_df_pq, use_container_width=True)

            st.subheader(f"Change in Price of Domestically Consumed Final Goods pq: {return_period}")
            st.bar_chart(app_df_pq, 
                         y="Change in pq (%)", 
                         x_label="Sectors",
                         y_label="Change in Price (%)",
                         height=400,
                         stack=False)
            
        except Exception as e:
            st.error(f"An error occurred: {e}")