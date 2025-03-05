document.addEventListener('DOMContentLoaded', function() {
    const factCheckForm = document.getElementById('fact-check-form');
    const contentGenForm = document.getElementById('content-gen-form');
    const factCheckResult = document.getElementById('fact-check-result');
    const contentGenResult = document.getElementById('content-gen-result');

    factCheckForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        factCheckResult.innerHTML = '<div class="loading">Processing...</div>';
        
        const content = document.getElementById('content').value;
        const researchDepth = document.getElementById('research-depth').value;
        
        try {
            const response = await fetch('/api/fact-check', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    content: content,
                    research_depth: researchDepth
                })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                factCheckResult.innerHTML = `
                    <div class="alert alert-info">
                        <div class="score-display">
                            Veracity Score: ${data.veracity_score}%
                        </div>
                        <div class="details">
                            <h6>Details:</h6>
                            <p>${data.details}</p>
                        </div>
                    </div>
                `;
            } else {
                throw new Error(data.error || 'Failed to process request');
            }
        } catch (error) {
            factCheckResult.innerHTML = `
                <div class="alert alert-danger">
                    Error: ${error.message}
                </div>
            `;
        }
    });

    contentGenForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        contentGenResult.innerHTML = '<div class="loading">Generating content...</div>';
        
        const topic = document.getElementById('topic').value;
        const researchDepth = document.getElementById('gen-research-depth').value;
        
        try {
            const response = await fetch('/api/generate-content', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    topic: topic,
                    research_depth: researchDepth
                })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                contentGenResult.innerHTML = `
                    <div class="alert alert-success">
                        <h6>Generated Content:</h6>
                        <div class="generated-content">
                            ${data.content}
                        </div>
                        <div class="metadata mt-3">
                            <small>
                                Generated using ${data.metadata.model_used} model
                                at ${new Date(data.metadata.timestamp).toLocaleString()}
                            </small>
                        </div>
                    </div>
                `;
            } else {
                throw new Error(data.error || 'Failed to generate content');
            }
        } catch (error) {
            contentGenResult.innerHTML = `
                <div class="alert alert-danger">
                    Error: ${error.message}
                </div>
            `;
        }
    });
});
