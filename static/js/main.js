document.addEventListener('DOMContentLoaded', function() {
    const factCheckForm = document.getElementById('fact-check-form');
    const contentGenForm = document.getElementById('content-gen-form');
    const factCheckResult = document.getElementById('fact-check-result');
    const contentGenResult = document.getElementById('content-gen-result');
    const factCheckStatus = document.getElementById('fact-check-status');
    const contentGenStatus = document.getElementById('content-gen-status');

    factCheckForm.addEventListener('submit', async function(e) {
        e.preventDefault();

        factCheckStatus.classList.remove('d-none');
        factCheckResult.innerHTML = '';

        const content = document.getElementById('content').value;

        try {
            const response = await fetch('/api/fact-check', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    content: content
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
        } finally {
            factCheckStatus.classList.add('d-none');
        }
    });

    contentGenForm.addEventListener('submit', async function(e) {
        e.preventDefault();

        contentGenStatus.classList.remove('d-none');
        contentGenResult.innerHTML = '';

        const topic = document.getElementById('topic').value;

        try {
            const response = await fetch('/api/generate-content', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    topic: topic
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
                                Generated at ${new Date(data.metadata.timestamp).toLocaleString()}
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
        } finally {
            contentGenStatus.classList.add('d-none');
        }
    });

    const factualContentForm = document.getElementById('factual-content-form');
    const factualContentStatus = document.getElementById('factual-content-status');
    const factualContentResult = document.getElementById('factual-content-result');

    factualContentForm.addEventListener('submit', async function(e) {
        e.preventDefault();

        const topic = document.getElementById('factual-topic').value;
        factualContentStatus.classList.remove('d-none');
        factualContentResult.innerHTML = '';

        try {
            const response = await fetch('/api/generate-factual-content', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ topic: topic })
            });

            const data = await response.json();

            if (response.ok) {
                if (data.status === 'COMPLETE') {
                    factualContentResult.innerHTML = `
                        <div class="alert alert-success">
                            <div class="mb-3">
                                <h6>Generated Content (${data.word_count} words):</h6>
                                <div class="generated-content article-content">
                                    ${data.content}
                                </div>
                            </div>
                            <div class="mb-3">
                                <h6>Structure:</h6>
                                <div class="content-structure">
                                    ${data.structure}
                                </div>
                            </div>
                            <div class="mb-3">
                                <h6>Verification:</h6>
                                <div class="verification-details">
                                    <p>Accuracy Score: ${data.verification.score}%</p>
                                    ${data.verification.improvements ? 
                                        `<p>Improvements: ${data.verification.improvements}</p>` : ''}
                                    ${data.verification.citations.length ? `
                                        <p>Sources:</p>
                                        <ul>
                                            ${data.verification.citations.map(cite => 
                                                `<li>${cite}</li>`).join('')}
                                        </ul>
                                    ` : ''}
                                </div>
                            </div>
                            <div class="metadata mt-3">
                                <small>
                                    Generated at ${new Date(data.metadata.timestamp).toLocaleString()}
                                </small>
                                <button class="btn btn-sm btn-outline-secondary ms-2" onclick="navigator.clipboard.writeText(${JSON.stringify(data.content_markdown)})">
                                    Copy Markdown
                                </button>
                            </div>
                        </div>
                    `;
                } else if (data.status === 'FAILED') {
                    factualContentResult.innerHTML = `
                        <div class="alert alert-danger">
                            Error: ${data.error}
                        </div>
                    `;
                }
            } else {
                throw new Error(data.error || 'Failed to generate factual content');
            }
        } catch (error) {
            factualContentResult.innerHTML = `
                <div class="alert alert-danger">
                    Error: ${error.message}
                </div>
            `;
        } finally {
            factualContentStatus.classList.add('d-none');
        }
    });
});