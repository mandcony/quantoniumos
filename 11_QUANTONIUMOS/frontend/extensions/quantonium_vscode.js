/**
 * VS Code Extension Integration Script
 * Provides QuantoniumOS commands and functionality within VS Code
 */

const vscode = require('vscode');
const { exec } = require('child_process');
const path = require('path');
const fs = require('fs');

/**
 * Extension activation function
 * @param {vscode.ExtensionContext} context 
 */
function activate(context) {
    console.log('🌌 QuantoniumOS Extension is now active!');

    // Register commands
    registerCommands(context);
    
    // Initialize status bar
    initializeStatusBar(context);
    
    // Set up window management
    setupWindowManagement(context);
    
    // Register configuration
    setupConfiguration(context);
}

function registerCommands(context) {
    // Launch QuantoniumOS
    const launchCommand = vscode.commands.registerCommand('quantonium.launch', () => {
        launchQuantoniumOS();
    });

    // Launch specific apps
    const launchRFTCommand = vscode.commands.registerCommand('quantonium.launch.rft', () => {
        launchQuantumApp('rft_visualizer');
    });

    const launchCryptoCommand = vscode.commands.registerCommand('quantonium.launch.crypto', () => {
        launchQuantumApp('quantum_crypto');
    });

    const launchSimulatorCommand = vscode.commands.registerCommand('quantonium.launch.simulator', () => {
        launchQuantumApp('quantum_simulator');
    });

    const launchMonitorCommand = vscode.commands.registerCommand('quantonium.launch.monitor', () => {
        launchQuantumApp('system_monitor');
    });

    // Window management commands
    const cascadeCommand = vscode.commands.registerCommand('quantonium.windows.cascade', () => {
        executeWindowCommand('cascade');
    });

    const tileHorizontalCommand = vscode.commands.registerCommand('quantonium.windows.tile.horizontal', () => {
        executeWindowCommand('tile_horizontal');
    });

    const tileVerticalCommand = vscode.commands.registerCommand('quantonium.windows.tile.vertical', () => {
        executeWindowCommand('tile_vertical');
    });

    const saveSessionCommand = vscode.commands.registerCommand('quantonium.session.save', () => {
        executeWindowCommand('save_session');
    });

    const loadSessionCommand = vscode.commands.registerCommand('quantonium.session.load', () => {
        executeWindowCommand('load_session');
    });

    // Quantum analysis commands
    const analyzeQuantumCommand = vscode.commands.registerCommand('quantonium.analyze.quantum', () => {
        analyzeCurrentFile();
    });

    const validateCryptoCommand = vscode.commands.registerCommand('quantonium.validate.crypto', () => {
        validateCryptography();
    });

    // Development commands
    const createQuantumAppCommand = vscode.commands.registerCommand('quantonium.create.app', () => {
        createQuantumApp();
    });

    const runQuantumTestCommand = vscode.commands.registerCommand('quantonium.test.quantum', () => {
        runQuantumTests();
    });

    // Register all commands
    context.subscriptions.push(
        launchCommand,
        launchRFTCommand,
        launchCryptoCommand,
        launchSimulatorCommand,
        launchMonitorCommand,
        cascadeCommand,
        tileHorizontalCommand,
        tileVerticalCommand,
        saveSessionCommand,
        loadSessionCommand,
        analyzeQuantumCommand,
        validateCryptoCommand,
        createQuantumAppCommand,
        runQuantumTestCommand
    );
}

function initializeStatusBar(context) {
    // Create status bar item
    const statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
    statusBarItem.text = "🌌 QuantoniumOS";
    statusBarItem.tooltip = "Click to launch QuantoniumOS";
    statusBarItem.command = 'quantonium.launch';
    statusBarItem.show();
    
    context.subscriptions.push(statusBarItem);

    // Update status based on QuantoniumOS state
    setInterval(() => {
        checkQuantoniumOSStatus(statusBarItem);
    }, 5000);
}

function setupWindowManagement(context) {
    // Monitor active editor changes
    vscode.window.onDidChangeActiveTextEditor(editor => {
        if (editor) {
            notifyQuantoniumOS('editor_changed', {
                file: editor.document.fileName,
                language: editor.document.languageId
            });
        }
    });

    // Monitor workspace changes
    vscode.workspace.onDidChangeWorkspaceFolders(event => {
        notifyQuantoniumOS('workspace_changed', {
            added: event.added.map(folder => folder.uri.fsPath),
            removed: event.removed.map(folder => folder.uri.fsPath)
        });
    });

    // Monitor configuration changes
    vscode.workspace.onDidChangeConfiguration(event => {
        if (event.affectsConfiguration('quantonium')) {
            reloadConfiguration();
        }
    });
}

function setupConfiguration(context) {
    // Get configuration
    const config = vscode.workspace.getConfiguration('quantonium');
    
    // Set up file watchers for quantum files
    const quantumFileWatcher = vscode.workspace.createFileSystemWatcher(
        '**/*.{py,cpp,ipynb}',
        false, // ignoreCreateEvents
        false, // ignoreChangeEvents
        false  // ignoreDeleteEvents
    );

    quantumFileWatcher.onDidCreate(uri => {
        analyzeQuantumFile(uri.fsPath);
    });

    quantumFileWatcher.onDidChange(uri => {
        if (config.get('autoAnalyze', true)) {
            analyzeQuantumFile(uri.fsPath);
        }
    });

    context.subscriptions.push(quantumFileWatcher);
}

function launchQuantoniumOS() {
    vscode.window.showInformationMessage('🚀 Launching QuantoniumOS...');
    
    const workspaceRoot = getWorkspaceRoot();
    if (!workspaceRoot) {
        vscode.window.showErrorMessage('❌ No workspace folder found');
        return;
    }

    const pythonPath = getPythonPath();
    const quantoniumScript = path.join(workspaceRoot, 'frontend', 'ui', 'quantum_app_controller.py');

    // Check if script exists
    if (!fs.existsSync(quantoniumScript)) {
        vscode.window.showErrorMessage(`❌ QuantoniumOS script not found: ${quantoniumScript}`);
        return;
    }

    // Launch QuantoniumOS
    exec(`"${pythonPath}" "${quantoniumScript}"`, { 
        cwd: workspaceRoot,
        env: { ...process.env, PYTHONPATH: workspaceRoot }
    }, (error, stdout, stderr) => {
        if (error) {
            vscode.window.showErrorMessage(`❌ Error launching QuantoniumOS: ${error.message}`);
            console.error('QuantoniumOS launch error:', error);
            return;
        }
        
        if (stderr) {
            console.warn('QuantoniumOS stderr:', stderr);
        }
        
        console.log('QuantoniumOS output:', stdout);
    });

    vscode.window.showInformationMessage('✅ QuantoniumOS launched successfully!');
}

function launchQuantumApp(appName) {
    vscode.window.showInformationMessage(`🚀 Launching ${appName}...`);
    
    // Send command to QuantoniumOS instance
    const command = {
        action: 'launch_app',
        app_name: appName,
        timestamp: new Date().toISOString()
    };

    sendCommandToQuantoniumOS(command);
}

function executeWindowCommand(commandName) {
    const command = {
        action: 'window_command',
        command: commandName,
        timestamp: new Date().toISOString()
    };

    sendCommandToQuantoniumOS(command);
}

function analyzeCurrentFile() {
    const activeEditor = vscode.window.activeTextEditor;
    if (!activeEditor) {
        vscode.window.showWarningMessage('⚠️ No active file to analyze');
        return;
    }

    const filePath = activeEditor.document.fileName;
    const fileContent = activeEditor.document.getText();

    vscode.window.showInformationMessage('🔬 Analyzing quantum properties...');

    // Check for quantum-related content
    const quantumPatterns = [
        /quantum/i,
        /qubit/i,
        /superposition/i,
        /entanglement/i,
        /rft|transform/i,
        /cryptography/i,
        /tensor/i,
        /amplitude/i
    ];

    const results = {
        file: filePath,
        isQuantumRelated: false,
        patterns: [],
        recommendations: []
    };

    quantumPatterns.forEach(pattern => {
        if (pattern.test(fileContent)) {
            results.isQuantumRelated = true;
            results.patterns.push(pattern.source);
        }
    });

    if (results.isQuantumRelated) {
        results.recommendations.push('Consider using QuantoniumOS quantum tools');
        results.recommendations.push('Validate with quantum cryptography module');
        
        // Show results in QuantoniumOS
        const command = {
            action: 'show_analysis',
            data: results,
            timestamp: new Date().toISOString()
        };
        
        sendCommandToQuantoniumOS(command);
        
        vscode.window.showInformationMessage('✅ Quantum analysis complete - view in QuantoniumOS');
    } else {
        vscode.window.showInformationMessage('ℹ️ No quantum patterns detected in current file');
    }
}

function validateCryptography() {
    const activeEditor = vscode.window.activeTextEditor;
    if (!activeEditor) {
        vscode.window.showWarningMessage('⚠️ No active file to validate');
        return;
    }

    vscode.window.showInformationMessage('🔐 Validating cryptographic implementation...');

    const command = {
        action: 'validate_crypto',
        file: activeEditor.document.fileName,
        content: activeEditor.document.getText(),
        timestamp: new Date().toISOString()
    };

    sendCommandToQuantoniumOS(command);
}

function createQuantumApp() {
    vscode.window.showInputBox({
        prompt: 'Enter quantum application name',
        placeHolder: 'my_quantum_app'
    }).then(appName => {
        if (appName) {
            const command = {
                action: 'create_app',
                app_name: appName,
                template: 'quantum_basic',
                timestamp: new Date().toISOString()
            };

            sendCommandToQuantoniumOS(command);
            vscode.window.showInformationMessage(`🌌 Creating quantum app: ${appName}`);
        }
    });
}

function runQuantumTests() {
    vscode.window.showInformationMessage('🧪 Running quantum tests...');
    
    const workspaceRoot = getWorkspaceRoot();
    if (!workspaceRoot) {
        vscode.window.showErrorMessage('❌ No workspace found');
        return;
    }

    const pythonPath = getPythonPath();
    
    // Run quantum validation tests
    exec(`"${pythonPath}" -m pytest tests/ -v -k quantum`, {
        cwd: workspaceRoot
    }, (error, stdout, stderr) => {
        if (error) {
            vscode.window.showErrorMessage(`❌ Test execution failed: ${error.message}`);
            return;
        }

        // Show results in output channel
        const outputChannel = vscode.window.createOutputChannel('QuantoniumOS Tests');
        outputChannel.clear();
        outputChannel.appendLine('🧪 Quantum Test Results');
        outputChannel.appendLine('==================');
        outputChannel.appendLine(stdout);
        
        if (stderr) {
            outputChannel.appendLine('\nErrors:');
            outputChannel.appendLine(stderr);
        }
        
        outputChannel.show();
        vscode.window.showInformationMessage('✅ Quantum tests completed - check output');
    });
}

function sendCommandToQuantoniumOS(command) {
    // Try to communicate with QuantoniumOS via file-based messaging
    const workspaceRoot = getWorkspaceRoot();
    if (!workspaceRoot) return;

    const commandFile = path.join(workspaceRoot, '.quantonium', 'vscode_commands.json');
    
    // Ensure directory exists
    const commandDir = path.dirname(commandFile);
    if (!fs.existsSync(commandDir)) {
        fs.mkdirSync(commandDir, { recursive: true });
    }

    // Write command
    try {
        fs.writeFileSync(commandFile, JSON.stringify(command, null, 2));
        console.log('Command sent to QuantoniumOS:', command);
    } catch (error) {
        console.error('Failed to send command:', error);
        vscode.window.showErrorMessage(`❌ Failed to communicate with QuantoniumOS: ${error.message}`);
    }
}

function notifyQuantoniumOS(eventType, data) {
    const command = {
        action: 'vscode_event',
        event_type: eventType,
        data: data,
        timestamp: new Date().toISOString()
    };

    sendCommandToQuantoniumOS(command);
}

function checkQuantoniumOSStatus(statusBarItem) {
    const workspaceRoot = getWorkspaceRoot();
    if (!workspaceRoot) return;

    const statusFile = path.join(workspaceRoot, '.quantonium', 'status.json');
    
    try {
        if (fs.existsSync(statusFile)) {
            const status = JSON.parse(fs.readFileSync(statusFile, 'utf8'));
            if (status.running) {
                statusBarItem.text = `🌌 QuantoniumOS (${status.windows} windows)`;
                statusBarItem.color = '#00ff00';
            } else {
                statusBarItem.text = '🌌 QuantoniumOS (stopped)';
                statusBarItem.color = '#ffff00';
            }
        } else {
            statusBarItem.text = '🌌 QuantoniumOS (not running)';
            statusBarItem.color = '#ff0000';
        }
    } catch (error) {
        console.error('Error checking QuantoniumOS status:', error);
    }
}

function analyzeQuantumFile(filePath) {
    console.log(`Analyzing quantum file: ${filePath}`);
    
    // Basic quantum file analysis
    try {
        const content = fs.readFileSync(filePath, 'utf8');
        const hasQuantumContent = /quantum|qubit|superposition|entanglement/i.test(content);
        
        if (hasQuantumContent) {
            const command = {
                action: 'file_analysis',
                file: filePath,
                quantum_score: calculateQuantumScore(content),
                timestamp: new Date().toISOString()
            };
            
            sendCommandToQuantoniumOS(command);
        }
    } catch (error) {
        console.error('Error analyzing file:', error);
    }
}

function calculateQuantumScore(content) {
    const quantumKeywords = [
        'quantum', 'qubit', 'superposition', 'entanglement', 
        'coherence', 'decoherence', 'amplitude', 'phase',
        'tensor', 'hilbert', 'observable', 'measurement',
        'rft', 'transform', 'encryption', 'cryptography'
    ];
    
    let score = 0;
    quantumKeywords.forEach(keyword => {
        const regex = new RegExp(keyword, 'gi');
        const matches = content.match(regex);
        if (matches) {
            score += matches.length;
        }
    });
    
    return Math.min(100, score * 2); // Cap at 100
}

function reloadConfiguration() {
    const config = vscode.workspace.getConfiguration('quantonium');
    console.log('QuantoniumOS configuration reloaded:', config);
    
    // Notify QuantoniumOS of configuration changes
    const command = {
        action: 'config_changed',
        config: {
            autoAnalyze: config.get('autoAnalyze', true),
            windowManagement: config.get('windowManagement', true),
            quantumValidation: config.get('quantumValidation', true)
        },
        timestamp: new Date().toISOString()
    };
    
    sendCommandToQuantoniumOS(command);
}

function getWorkspaceRoot() {
    const workspaceFolders = vscode.workspace.workspaceFolders;
    return workspaceFolders && workspaceFolders.length > 0 
        ? workspaceFolders[0].uri.fsPath 
        : null;
}

function getPythonPath() {
    // Try to get Python path from VS Code Python extension
    const pythonExtension = vscode.extensions.getExtension('ms-python.python');
    if (pythonExtension && pythonExtension.isActive) {
        try {
            const pythonPath = pythonExtension.exports.settings.getExecutionDetails().execCommand[0];
            if (pythonPath) return pythonPath;
        } catch (error) {
            console.warn('Could not get Python path from extension:', error);
        }
    }

    // Fallback to system Python
    return process.platform === 'win32' ? 'python' : 'python3';
}

function deactivate() {
    console.log('🌌 QuantoniumOS Extension deactivated');
}

module.exports = {
    activate,
    deactivate
};
