from flask import Flask, render_template, jsonify, request
from engine import engine, ForexEngine
from live_predict import live_predictor
import threading
import os

app = Flask(__name__)

# Other API routes...

# Global state
import time

AUTO_REFRESH = True
REFRESH_INTERVAL = 60 * 60      # 1 hour

pipeline_status = {
    'running': False, 
    'results': None
    }


def run_pipeline_async():
    """Run the ML pipeline in background thread"""
    global pipeline_status
    pipeline_status['running'] = True
    pipeline_status['results'] = None
    
    try:
        results = engine.run_pipeline()
        pipeline_status['results'] = results
    except Exception as e:
        pipeline_status['results'] = {
            'status': 'error',
            'error': str(e),
            'steps': [{'name': 'Error', 'message': str(e), 'status': 'error'}]
        }
    finally:
        pipeline_status['running'] = False


def auto_refresh_pipeline():
    """Automatically refresh the ML pipeline every hour."""
    while AUTO_REFRESH:
        try:
            if not pipeline_status['running']:
                print("Auto-refresh: Running scheduled analysis...")
                run_pipeline_async()
        except Exception as e:
            print(f"Auto-refresh error: {e}")

        time.sleep(REFRESH_INTERVAL)

@app.route('/')
def landing():
    return render_template('landing.html')


@app.route('/dashboard')
def dashboard():
    return render_template('index.html')

@app.route('/api/run', methods=['POST'])
def run_analysis():
    """Start the ML pipeline"""
    if pipeline_status['running']:
        return jsonify({'status': 'error', 'message': 'Pipeline already running'})

    thread = threading.Thread(target=run_pipeline_async)
    thread.daemon = True
    thread.start()
    return jsonify({'status': 'started', 'message': 'Pipeline started'})

@app.route('/api/status', methods=['GET'])
def get_status():

    response = {
        'running': pipeline_status['running'],
        'results': pipeline_status['results']
    }

    try:
        live = live_predictor.predict()

        if pipeline_status['results'] is not None:
            pipeline_status['results']['live_prediction'] = live

        response['live_prediction'] = live

    except Exception as e:
        response['live_prediction'] = {
            "error": str(e)
        }

    return jsonify(response)

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': pd.Timestamp.now().isoformat()})

# Import here to avoid circular import
import pandas as pd

print("\nRegistered Routes:")
for rule in app.url_map.iter_rules():
    print(rule)
print()

if __name__ == "__main__":

    # Start automatic background refresh
    refresh_thread = threading.Thread(target=auto_refresh_pipeline)
    refresh_thread.daemon = True
    refresh_thread.start()

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=True,
        use_reloader=False
    )