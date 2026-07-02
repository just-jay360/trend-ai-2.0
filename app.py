from flask import Flask, render_template, jsonify, request
from engine import engine, ForexEngine
import threading

app = Flask(__name__)

# Global state
pipeline_status = {'running': False, 'results': None}

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

@app.route('/')
def dashboard():
    """Main dashboard page"""
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
    """Check pipeline status"""
    return jsonify({
        'running': pipeline_status['running'],
        'results': pipeline_status['results']
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': pd.Timestamp.now().isoformat()})

# Import here to avoid circular import
import pandas as pd

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))