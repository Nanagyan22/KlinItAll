import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import math
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Geospatial", page_icon="🌍", layout="wide")

st.title("🌍 Geospatial")
st.markdown("Advanced geographic data validation, analysis, and visualization capabilities")

if 'current_dataset' not in st.session_state or st.session_state.current_dataset is None:
    st.warning("⚠️ No dataset found. Please upload data first.")
    if st.button("📥 Go to Upload Page"):
        st.switch_page("pages/01_Upload.py")
    st.stop()

df = st.session_state.current_dataset.copy()

# Detect potential geospatial columns
def detect_geo_columns(df):
    """Detect potential geospatial columns"""
    geo_keywords = {
        'latitude': ['lat', 'latitude', 'y_coord', 'y'],
        'longitude': ['lon', 'lng', 'longitude', 'x_coord', 'x'],
        'address': ['address', 'addr', 'location', 'place'],
        'city': ['city', 'town', 'municipality'],
        'state': ['state', 'province', 'region'],
        'country': ['country', 'nation'],
        'zipcode': ['zip', 'zipcode', 'postal', 'postcode']
    }
    
    detected = {}
    
    for col in df.columns:
        col_lower = col.lower()
        for geo_type, keywords in geo_keywords.items():
            if any(keyword in col_lower for keyword in keywords):
                if geo_type not in detected:
                    detected[geo_type] = []
                detected[geo_type].append(col)
    
    return detected

# Geospatial utility functions
def validate_coordinates(lat, lon):
    """Validate latitude and longitude coordinates"""
    try:
        lat_val = float(lat)
        lon_val = float(lon)
        
        lat_valid = -90 <= lat_val <= 90
        lon_valid = -180 <= lon_val <= 180
        
        return lat_valid and lon_valid
    except:
        return False

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points using Haversine formula"""
    try:
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Earth's radius in kilometers
        r = 6371
        
        return c * r
    except:
        return np.nan

# Detect geospatial columns
geo_columns = detect_geo_columns(df)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Find coordinate columns
lat_candidates = [col for col in numeric_cols if any(keyword in col.lower() for keyword in ['lat', 'latitude', 'y'])]
lon_candidates = [col for col in numeric_cols if any(keyword in col.lower() for keyword in ['lon', 'lng', 'longitude', 'x'])]

# Overview metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Detected Geo Types", len(geo_columns))

with col2:
    lat_cols = len(lat_candidates)
    st.metric("Latitude Candidates", lat_cols)

with col3:
    lon_cols = len(lon_candidates)
    st.metric("Longitude Candidates", lon_cols)

with col4:
    if lat_candidates and lon_candidates:
        st.metric("Coordinate Pairs", "Available")
    else:
        st.metric("Coordinate Pairs", "None")

# Geospatial tabs
geo_tabs = st.tabs(["🔍 Detection", "✅ Validation", "📊 Analysis", "🗺️ Visualization"])

with geo_tabs[0]:
    st.markdown("### 🔍 Geospatial Column Detection")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### Detected Geospatial Columns")
        
        if geo_columns:
            for geo_type, columns in geo_columns.items():
                st.markdown(f"**{geo_type.title()}:** {', '.join(columns)}")
        else:
            st.info("No geospatial columns auto-detected.")
        
        # Manual column assignment
        st.markdown("#### Manual Column Assignment")
        
        manual_assignment = {}
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            manual_assignment['latitude'] = st.selectbox(
                "Latitude column:",
                ['None'] + df.columns.tolist(),
                help="Select the column containing latitude values"
            )
            
            manual_assignment['longitude'] = st.selectbox(
                "Longitude column:",
                ['None'] + df.columns.tolist(),
                help="Select the column containing longitude values"
            )
            
            manual_assignment['address'] = st.selectbox(
                "Address column:",
                ['None'] + df.columns.tolist(),
                help="Select the column containing address information"
            )
        
        with col_b:
            manual_assignment['city'] = st.selectbox(
                "City column:",
                ['None'] + df.columns.tolist(),
                help="Select the column containing city names"
            )
            
            manual_assignment['state'] = st.selectbox(
                "State/Region column:",
                ['None'] + df.columns.tolist(),
                help="Select the column containing state or region"
            )
            
            manual_assignment['country'] = st.selectbox(
                "Country column:",
                ['None'] + df.columns.tolist(),
                help="Select the column containing country information"
            )
        
        # Update geo_columns with manual assignments
        for geo_type, col_name in manual_assignment.items():
            if col_name != 'None':
                if geo_type not in geo_columns:
                    geo_columns[geo_type] = []
                if col_name not in geo_columns[geo_type]:
                    geo_columns[geo_type].append(col_name)
    
    with col2:
        st.markdown("#### Column Samples")
        
        if geo_columns:
            sample_geo_type = st.selectbox("View samples for:", list(geo_columns.keys()))
            
            if sample_geo_type and geo_columns[sample_geo_type]:
                sample_col = geo_columns[sample_geo_type][0]
                sample_data = df[sample_col].dropna().head(10)
                
                st.markdown(f"**Sample values from {sample_col}:**")
                for i, val in enumerate(sample_data, 1):
                    st.write(f"{i}. {val}")

with geo_tabs[1]:
    st.markdown("### ✅ Geospatial Data Validation")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Coordinate Validation")
        
        if 'latitude' in geo_columns and 'longitude' in geo_columns:
            lat_col = st.selectbox("Select latitude column:", geo_columns['latitude'], key="val_lat")
            lon_col = st.selectbox("Select longitude column:", geo_columns['longitude'], key="val_lon")
            
            if st.button("🔍 Validate Coordinates", type="primary"):
                try:
                    # Validate coordinates
                    valid_coords = []
                    invalid_coords = []
                    
                    for idx, row in df.iterrows():
                        lat_val = row[lat_col]
                        lon_val = row[lon_col]
                        
                        if pd.notna(lat_val) and pd.notna(lon_val):
                            if validate_coordinates(lat_val, lon_val):
                                valid_coords.append(idx)
                            else:
                                invalid_coords.append(idx)
                    
                    # Results
                    total_coords = len(valid_coords) + len(invalid_coords)
                    valid_percentage = (len(valid_coords) / total_coords * 100) if total_coords > 0 else 0
                    
                    st.markdown("#### Validation Results")
                    
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        st.metric("Valid Coordinates", f"{len(valid_coords):,}")
                    with col_b:
                        st.metric("Invalid Coordinates", f"{len(invalid_coords):,}")
                    with col_c:
                        st.metric("Success Rate", f"{valid_percentage:.1f}%")
                    
                    # Show invalid coordinates
                    if invalid_coords:
                        st.markdown("#### Sample Invalid Coordinates")
                        invalid_sample = df.loc[invalid_coords[:5], [lat_col, lon_col]]
                        
                        for col in invalid_sample.columns:
                            invalid_sample[col] = invalid_sample[col].astype(str)
                        
                        st.dataframe(invalid_sample, use_container_width=True)
                        
                        # Option to fix invalid coordinates
                        if st.button("🔧 Handle Invalid Coordinates"):
                            fix_method = st.selectbox(
                                "How to handle invalid coordinates:",
                                ["Remove rows with invalid coordinates", "Set invalid coordinates to NaN", "Flag as invalid"]
                            )
                            
                            if st.button("Apply Fix", key="apply_coord_fix"):
                                if fix_method == "Remove rows with invalid coordinates":
                                    df = df.drop(index=invalid_coords)
                                    st.success(f"✅ Removed {len(invalid_coords)} rows with invalid coordinates")
                                
                                elif fix_method == "Set invalid coordinates to NaN":
                                    df.loc[invalid_coords, [lat_col, lon_col]] = np.nan
                                    st.success(f"✅ Set {len(invalid_coords)} invalid coordinate pairs to NaN")
                                
                                elif fix_method == "Flag as invalid":
                                    df['coordinate_valid'] = True
                                    df.loc[invalid_coords, 'coordinate_valid'] = False
                                    st.success(f"✅ Flagged {len(invalid_coords)} rows with invalid coordinates")
                                
                                st.session_state.current_dataset = df
                                st.rerun()
                    
                    else:
                        st.success("✅ All coordinates are valid!")
                    
                except Exception as e:
                    st.error(f"❌ Validation failed: {str(e)}")
        
        else:
            st.info("Please assign latitude and longitude columns in the Detection tab first.")
    
    with col2:
        st.markdown("#### Range Analysis")
        
        if lat_candidates and lon_candidates:
            analysis_lat = st.selectbox("Latitude column for analysis:", lat_candidates, key="analysis_lat")
            analysis_lon = st.selectbox("Longitude column for analysis:", lon_candidates, key="analysis_lon")
            
            lat_data = df[analysis_lat].dropna()
            lon_data = df[analysis_lon].dropna()
            
            if len(lat_data) > 0 and len(lon_data) > 0:
                # Coordinate statistics
                coord_stats = {
                    'Metric': ['Latitude Min', 'Latitude Max', 'Latitude Range', 'Longitude Min', 'Longitude Max', 'Longitude Range'],
                    'Value': [
                        f"{lat_data.min():.6f}",
                        f"{lat_data.max():.6f}",
                        f"{lat_data.max() - lat_data.min():.6f}",
                        f"{lon_data.min():.6f}",
                        f"{lon_data.max():.6f}",
                        f"{lon_data.max() - lon_data.min():.6f}"
                    ]
                }
                
                stats_df = pd.DataFrame(coord_stats)
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
                
                # Geographic bounds estimation
                if lat_data.min() >= -90 and lat_data.max() <= 90 and lon_data.min() >= -180 and lon_data.max() <= 180:
                    # Estimate geographic region
                    lat_center = (lat_data.min() + lat_data.max()) / 2
                    lon_center = (lon_data.min() + lon_data.max()) / 2
                    
                    # Simple region estimation based on coordinates
                    if -48 <= lat_center <= 71 and -168 <= lon_center <= -52:
                        region = "North America"
                    elif 35 <= lat_center <= 71 and -12 <= lon_center <= 40:
                        region = "Europe"
                    elif -35 <= lat_center <= 37 and -20 <= lon_center <= 55:
                        region = "Africa"
                    elif -47 <= lat_center <= 81 and 26 <= lon_center <= 180:
                        region = "Asia"
                    else:
                        region = "Mixed/Global"
                    
                    st.markdown(f"**Estimated Region:** {region}")
                    st.markdown(f"**Center Point:** ({lat_center:.4f}, {lon_center:.4f})")

with geo_tabs[2]:
    st.markdown("### 📊 Geospatial Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Distance Analysis")
        
        if lat_candidates and lon_candidates:
            dist_lat = st.selectbox("Latitude column:", lat_candidates, key="dist_lat")
            dist_lon = st.selectbox("Longitude column:", lon_candidates, key="dist_lon")
            
            # Reference point for distance calculations
            ref_method = st.selectbox(
                "Reference point method:",
                ["Dataset centroid", "Custom coordinates", "First data point"],
                help="Choose reference point for distance calculations"
            )
            
            if ref_method == "Custom coordinates":
                ref_lat = st.number_input("Reference latitude:", value=0.0, format="%.6f")
                ref_lon = st.number_input("Reference longitude:", value=0.0, format="%.6f")
            
            if st.button("📏 Calculate Distances", type="primary"):
                try:
                    # Get clean coordinate data
                    coord_data = df[[dist_lat, dist_lon]].dropna()
                    
                    if len(coord_data) == 0:
                        st.error("No valid coordinate pairs found.")
                        st.stop()
                    
                    # Determine reference point
                    if ref_method == "Dataset centroid":
                        ref_lat = coord_data[dist_lat].mean()
                        ref_lon = coord_data[dist_lon].mean()
                    elif ref_method == "First data point":
                        ref_lat = coord_data[dist_lat].iloc[0]
                        ref_lon = coord_data[dist_lon].iloc[0]
                    
                    # Calculate distances
                    distances = []
                    for _, row in coord_data.iterrows():
                        dist = calculate_distance(ref_lat, ref_lon, row[dist_lat], row[dist_lon])
                        distances.append(dist)
                    
                    # Add distances to dataframe
                    distance_col_name = f"distance_from_reference_km"
                    coord_data[distance_col_name] = distances
                    
                    # Update original dataframe
                    df = df.merge(coord_data[[distance_col_name]], left_index=True, right_index=True, how='left')
                    st.session_state.current_dataset = df
                    
                    # Distance statistics
                    dist_stats = {
                        'Metric': ['Mean Distance', 'Median Distance', 'Min Distance', 'Max Distance', 'Std Deviation'],
                        'Value (km)': [
                            f"{np.mean(distances):.2f}",
                            f"{np.median(distances):.2f}",
                            f"{np.min(distances):.2f}",
                            f"{np.max(distances):.2f}",
                            f"{np.std(distances):.2f}"
                        ]
                    }
                    
                    dist_stats_df = pd.DataFrame(dist_stats)
                    st.dataframe(dist_stats_df, use_container_width=True, hide_index=True)
                    
                    st.success(f"✅ Added distance calculations! Reference: ({ref_lat:.4f}, {ref_lon:.4f})")
                    
                    # Log action
                    if 'processing_log' not in st.session_state:
                        st.session_state.processing_log = []
                    
                    st.session_state.processing_log.append({
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'action': 'Geospatial Analysis',
                        'details': f"Calculated distances from reference point ({ref_lat:.4f}, {ref_lon:.4f}) for {len(distances)} coordinates"
                    })
                    
                except Exception as e:
                    st.error(f"❌ Distance calculation failed: {str(e)}")
    
    with col2:
        st.markdown("#### Density Analysis")
        
        if lat_candidates and lon_candidates:
            density_lat = st.selectbox("Latitude column:", lat_candidates, key="density_lat")
            density_lon = st.selectbox("Longitude column:", lon_candidates, key="density_lon")
            
            if st.button("🗺️ Analyze Point Density"):
                try:
                    coord_data = df[[density_lat, density_lon]].dropna()
                    
                    if len(coord_data) < 2:
                        st.error("Need at least 2 valid coordinate pairs for density analysis.")
                        st.stop()
                    
                    # Simple density analysis using coordinate clustering
                    # Divide space into grid cells and count points
                    lat_min, lat_max = coord_data[density_lat].min(), coord_data[density_lat].max()
                    lon_min, lon_max = coord_data[density_lon].min(), coord_data[density_lon].max()
                    
                    # Grid resolution (adjust based on data range)
                    lat_range = lat_max - lat_min
                    lon_range = lon_max - lon_min
                    
                    if lat_range > 0 and lon_range > 0:
                        grid_size = min(10, max(3, int(len(coord_data) ** 0.5)))  # Adaptive grid size
                        
                        lat_bins = np.linspace(lat_min, lat_max, grid_size + 1)
                        lon_bins = np.linspace(lon_min, lon_max, grid_size + 1)
                        
                        # Count points in each grid cell
                        density_counts = []
                        grid_coords = []
                        
                        for i in range(grid_size):
                            for j in range(grid_size):
                                lat_low, lat_high = lat_bins[i], lat_bins[i + 1]
                                lon_low, lon_high = lon_bins[j], lon_bins[j + 1]
                                
                                points_in_cell = len(coord_data[
                                    (coord_data[density_lat] >= lat_low) & 
                                    (coord_data[density_lat] < lat_high) &
                                    (coord_data[density_lon] >= lon_low) & 
                                    (coord_data[density_lon] < lon_high)
                                ])
                                
                                if points_in_cell > 0:
                                    density_counts.append(points_in_cell)
                                    grid_coords.append({
                                        'lat_center': (lat_low + lat_high) / 2,
                                        'lon_center': (lon_low + lon_high) / 2,
                                        'count': points_in_cell
                                    })
                        
                        # Density statistics
                        if density_counts:
                            st.markdown("#### Density Statistics")
                            st.write(f"**Grid Cells with Data:** {len(density_counts)}")
                            st.write(f"**Max Points per Cell:** {max(density_counts)}")
                            st.write(f"**Avg Points per Cell:** {np.mean(density_counts):.1f}")
                            st.write(f"**Total Points Analyzed:** {len(coord_data)}")
                            
                            # Find highest density area
                            max_density_cell = max(grid_coords, key=lambda x: x['count'])
                            st.write(f"**Highest Density Location:** ({max_density_cell['lat_center']:.4f}, {max_density_cell['lon_center']:.4f}) with {max_density_cell['count']} points")
                        
                        else:
                            st.info("Points are too dispersed for meaningful density analysis with current grid resolution.")
                    
                    else:
                        st.info("Coordinate range too small for density analysis.")
                        
                except Exception as e:
                    st.error(f"❌ Density analysis failed: {str(e)}")

with geo_tabs[3]:
    st.markdown("### 🗺️ Geospatial Visualization")
    
    if lat_candidates and lon_candidates:
        viz_lat = st.selectbox("Latitude column for visualization:", lat_candidates, key="viz_lat")
        viz_lon = st.selectbox("Longitude column for visualization:", lon_candidates, key="viz_lon")
        
        # Optional color/size columns
        numeric_cols_viz = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            color_col = st.selectbox(
                "Color by column (optional):",
                ['None'] + categorical_cols + numeric_cols_viz,
                help="Choose a column to color-code the points"
            )
            
            size_col = st.selectbox(
                "Size by column (optional):",
                ['None'] + numeric_cols_viz,
                help="Choose a numeric column to size the points"
            )
        
        with col2:
            map_style = st.selectbox(
                "Map visualization:",
                ["Scatter plot", "Density heatmap", "Simple map"],
                help="Choose the type of visualization"
            )
            max_points = st.slider(
                "Maximum points to display:",
                min_value=1,
                max_value=max(len(df), 2),  # ensures max > min
                value=min(1000, len(df)),  # safe default
                help="Limit points for performance"
            )

            #max_points = st.slider(
                #"Maximum points to display:",
                #100, min(5000, len(df)), min(1000, len(df)),
                #help="Limit points for performance"
            #)
        
        if st.button("🗺️ Create Visualization", type="primary"):
            try:
                # Get coordinate data
                viz_data = df[[viz_lat, viz_lon]].dropna()
                
                # Add additional columns if specified
                extra_cols = []
                if color_col != 'None':
                    extra_cols.append(color_col)
                if size_col != 'None':
                    extra_cols.append(size_col)
                
                if extra_cols:
                    viz_data = df[[viz_lat, viz_lon] + extra_cols].dropna()
                
                # Limit points for performance
                if len(viz_data) > max_points:
                    viz_data = viz_data.sample(n=max_points)
                    st.info(f"Displaying {max_points:,} randomly sampled points out of {len(df):,} total points.")
                
                if len(viz_data) == 0:
                    st.error("No valid coordinate data found for visualization.")
                    st.stop()
                
                # Create visualization based on selected type
                if map_style == "Scatter plot":
                    fig = px.scatter(
                        viz_data, 
                        x=viz_lon, 
                        y=viz_lat,
                        color=color_col if color_col != 'None' else None,
                        size=size_col if size_col != 'None' else None,
                        title=f"Geographic Scatter Plot ({len(viz_data):,} points)",
                        labels={viz_lat: "Latitude", viz_lon: "Longitude"}
                    )
                    
                    fig.update_layout(
                        xaxis_title="Longitude",
                        yaxis_title="Latitude",
                        height=600
                    )
                
                elif map_style == "Density heatmap":
                    fig = px.density_heatmap(
                        viz_data,
                        x=viz_lon,
                        y=viz_lat,
                        title=f"Geographic Density Heatmap ({len(viz_data):,} points)",
                        labels={viz_lat: "Latitude", viz_lon: "Longitude"}
                    )
                    
                    fig.update_layout(height=600)
                
                else:  # Simple map
                    # For simple map, we'll create a scatter plot with map-like styling
                    fig = go.Figure()
                    
                    if color_col != 'None':
                        # Color-coded points
                        unique_values = viz_data[color_col].unique()
                        colors = px.colors.qualitative.Set3[:len(unique_values)]
                        
                        for i, value in enumerate(unique_values):
                            subset = viz_data[viz_data[color_col] == value]
                            fig.add_trace(go.Scatter(
                                x=subset[viz_lon],
                                y=subset[viz_lat],
                                mode='markers',
                                name=str(value),
                                marker=dict(
                                    color=colors[i % len(colors)],
                                    size=8 if size_col == 'None' else subset[size_col] * 2,
                                    opacity=0.7
                                )
                            ))
                    else:
                        # Single color points
                        fig.add_trace(go.Scatter(
                            x=viz_data[viz_lon],
                            y=viz_data[viz_lat],
                            mode='markers',
                            name='Data Points',
                            marker=dict(
                                color='blue',
                                size=8 if size_col == 'None' else viz_data[size_col] * 2,
                                opacity=0.7
                            )
                        ))
                    
                    fig.update_layout(
                        title=f"Geographic Distribution ({len(viz_data):,} points)",
                        xaxis_title="Longitude",
                        yaxis_title="Latitude",
                        height=600,
                        showlegend=color_col != 'None'
                    )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Basic geographic statistics
                st.markdown("#### Geographic Summary")
                
                geo_summary = {
                    'Center Latitude': f"{viz_data[viz_lat].mean():.6f}",
                    'Center Longitude': f"{viz_data[viz_lon].mean():.6f}",
                    'Latitude Range': f"{viz_data[viz_lat].max() - viz_data[viz_lat].min():.6f}",
                    'Longitude Range': f"{viz_data[viz_lon].max() - viz_data[viz_lon].min():.6f}",
                    'Bounding Box': f"({viz_data[viz_lat].min():.4f}, {viz_data[viz_lon].min():.4f}) to ({viz_data[viz_lat].max():.4f}, {viz_data[viz_lon].max():.4f})"
                }
                
                for key, value in geo_summary.items():
                    st.write(f"**{key}:** {value}")
                
            except Exception as e:
                st.error(f"❌ Visualization failed: {str(e)}")
    
    else:
        st.info("No coordinate columns available for visualization. Please check the Detection tab.")

# Export and Navigation
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("💾 Download Enhanced Dataset"):
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"geospatial_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

with col2:
    if st.button("🔍 View Data Overview"):
        st.switch_page("pages/02_Data_Overview.py")

with col3:
    if st.button("➡️ Continue to Categorical Encoding"):
        st.switch_page("pages/11_Categorical_Encoding.py")

# Sidebar
with st.sidebar:
    st.markdown("### 🌍 Geospatial Processing Guide")
    
    st.markdown("#### Key Features:")
    features = [
        "**Detection:** Auto-find geo columns",
        "**Validation:** Coordinate range checking",
        "**Analysis:** Distance & density calculations", 
        "**Visualization:** Maps and spatial plots"
    ]
    
    for feature in features:
        st.markdown(f"• {feature}")
    
    st.markdown("---")
    st.markdown("#### Coordinate Formats:")
    
    formats = [
        "Decimal degrees (41.8781, -87.6298)",
        "Valid latitude: -90 to +90",
        "Valid longitude: -180 to +180",
        "Common columns: lat, lon, x_coord, y_coord"
    ]
    
    for fmt in formats:
        st.markdown(f"• {fmt}")
    
    st.markdown("---")
    st.markdown("#### 💡 Best Practices")
    
    st.info("""
    **Guidelines:**
    • Validate coordinate ranges first
    • Check for coordinate system consistency
    • Consider map projections for analysis
    • Handle missing coordinates appropriately
    • Use appropriate visualization for data size
    """)