import React, { useState, useEffect, useRef, useCallback } from 'react';
import { io } from 'socket.io-client';
import { useNavigate, useLocation } from 'react-router-dom';
import Peer from 'simple-peer';
import { GestureRecognizer, FilesetResolver } from '@mediapipe/tasks-vision';
import '../style.css';

// ─── CONFIG ──────────────────────────────────────────────────────────────────
const SOCKET_URL = 'http://localhost:5000';

const ICE_CONFIG = {
  iceServers: [
    { urls: 'stun:stun.l.google.com:19302' },
    { urls: 'stun:stun1.l.google.com:19302' },
  ],
};

const GESTURE_ICONS = {
  'Closed_Fist':      '✊',
  'Open_Palm':        '🖐️',
  'Pointing_Up':      '☝️',
  'Thumb_Up':         '👍',
  'Thumb_Down':       '👎',
  'Victory':          '✌️',
  'ILoveYou':         '🤟',
  'Space':            '⏹️', 
  'None':             '🤚',
};

const GESTURE_TO_LETTER = {
  'Victory':     'V',
  'Pointing Up': 'I',
  'Thumb Up':    'A',
  'Thumb Down':  'B',
  'Closed Fist': 'E',
  'Iloveyou':    'L',
  'Open Palm':   'O'
};

const prettifyGesture = (raw) =>
  raw.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());

// ─── VideoNode Component ──────────────────────────────────────────────────────
const VideoNode = React.memo(({ peer, name, isSelected, onClick }) => {
  const vidRef = useRef();

  useEffect(() => {
    if (!peer) return;
    const attach = (stream) => {
      if (!vidRef.current || vidRef.current.srcObject === stream) return;
      vidRef.current.srcObject = stream;
      vidRef.current.play().catch(() => {});
    };
    if (peer.streams?.[0]) attach(peer.streams[0]);
    peer.on('stream', attach);
    return () => peer.off('stream', attach);
  }, [peer]);

  return (
    <div 
      className={`participant-card ${isSelected ? 'selected-focus-border' : 'active-speaker'}`}
      onClick={onClick}
      style={{ cursor: 'pointer' }}
    >
      <div className="remote-video-wrap">
        <video ref={vidRef} autoPlay playsInline />
      </div>
      <span className="participant-name">{name}</span>
      <div className="participant-status active" />
    </div>
  );
});

// ─── Main component ───────────────────────────────────────────────────────────
const VideoCall = () => {
  const navigate = useNavigate();
  const location = useLocation();

  const [currentRoomId] = useState(() => {
    const p = new URLSearchParams(location.search);
    return p.get('room') || 'RM-' + Math.random().toString(36).slice(2, 6).toUpperCase();
  });

  const [isRunning,      setIsRunning]      = useState(false);
  const [currentGesture, setCurrentGesture] = useState('None');
  const [confidence,     setConfidence]     = useState('—');
  const [historyFeed,    setHistoryFeed]    = useState([]);
  const [chatMessage,    setChatMessage]    = useState('');
  const [stats,          setStats]          = useState({ count: 0, totalConf: 0, freq: {} });
  const [activePeers,    setActivePeers]    = useState([]);
  const [linkCopied,     setLinkCopied]     = useState(false);
  const [modelStatus,    setModelStatus]    = useState('idle');
  
  const [accumulatedWord, setAccumulatedWord] = useState('');
  const [showWelcomeModal, setShowWelcomeModal] = useState(true);
  const [isMuted,        setIsMuted]        = useState(false);

  // 🌟 Theater Mode Layout Active Selection Tracking States
  const [selectedPeerId, setSelectedPeerId] = useState(null);

  const localVidRef      = useRef(null);
  const canvasRef        = useRef(null);
  const streamRef        = useRef(null);
  const socketRef        = useRef(null);
  const rafRef           = useRef(null);       
  const isRunningRef     = useRef(false);
  const historyRef       = useRef([]);
  const peersRef         = useRef(new Map());
  const gestureRef       = useRef(null);       
  const lastGestureRef   = useRef('None');
  const lastVideoTimeRef = useRef(-1);
  
  const lastProcessingTimeRef = useRef(0);
  const wordBufferRef          = useRef('');
  
  const gestureHistoryRef     = useRef([]);
  const lastAppendedSignRef   = useRef('');
  const appendCooldownRef      = useRef(0);

  // Big Grid Ref for Theater view mirroring
  const theaterVidRef        = useRef(null);

  const userName = localStorage.getItem('userName') || 'User';

  useEffect(() => { historyRef.current = historyFeed; }, [historyFeed]);

  // 🌟 Handle Mirroring the Active selected user into the main center screen
  useEffect(() => {
    if (!theaterVidRef.current) return;
    
    if (selectedPeerId) {
      const targetRecord = peersRef.current.get(selectedPeerId);
      const targetStream = targetRecord?.peer?.streams?.[0];
      if (targetStream) {
        theaterVidRef.current.srcObject = targetStream;
        theaterVidRef.current.play().catch(() => {});
        return;
      }
    }
    // Fallback to null if peer goes missing
    theaterVidRef.current.srcObject = null;
  }, [selectedPeerId, activePeers]);

  const speakText = (textToSpeak) => {
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel(); 
      const utterance = new SpeechSynthesisUtterance(textToSpeak);
      utterance.rate = 1.0;  
      utterance.pitch = 1.0; 
      window.speechSynthesis.speak(utterance);
    }
  };

  useEffect(() => {
    let cancelled = false;
    const loadModel = async () => {
      try {
        setModelStatus('loading');
        const vision = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
        );
        const recognizer = await GestureRecognizer.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: '/gesture_recognizer.task',
            delegate: 'GPU',   
          },
          runningMode: 'VIDEO',
          numHands: 2,
          minHandDetectionConfidence: 0.5,
          minHandPresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });

        if (cancelled) { recognizer.close(); return; }
        gestureRef.current = recognizer;
        setModelStatus('ready');
      } catch (err) {
        console.error('[MediaPipe] model load failed:', err);
        setModelStatus('error');
      }
    };
    loadModel();
    return () => { cancelled = true; };
  }, []);

  const dispatchSignWord = useCallback(() => {
    const finalWord = wordBufferRef.current.trim();
    if (!finalWord) return;

    const ts = new Date().toLocaleTimeString('en-US', { hour12: false });

    if (socketRef.current) {
      socketRef.current.emit('chatMessage', { 
        roomId: currentRoomId, 
        msg: `[Sign] ${finalWord}`, 
        senderName: userName 
      });
    }

    setHistoryFeed(prev => [
      { id: Date.now(), type: 'chat', icon: '🤟', text: `You (Sign): ${finalWord}`, meta: ts },
      ...prev.slice(0, 49)
    ]);

    speakText(finalWord);

    wordBufferRef.current = '';
    setAccumulatedWord('');
    lastAppendedSignRef.current = ''; 
  }, [currentRoomId, userName]);

  const processLiveGesture = useCallback((sign, confStr) => {
    const rawSign = sign.replace(/ /g, '_');
    
    if (rawSign.toLowerCase() === 'space') {
      dispatchSignWord();
      return;
    }

    if (rawSign === 'None') return;

    if (sign === lastAppendedSignRef.current && Date.now() - appendCooldownRef.current < 2000) {
      return; 
    }

    let characterToken = GESTURE_TO_LETTER[sign] || sign;
    if (characterToken.length > 2) {
      characterToken = characterToken.charAt(0).toUpperCase(); 
    }

    wordBufferRef.current += characterToken;
    setAccumulatedWord(wordBufferRef.current);
    
    lastAppendedSignRef.current = sign;
    appendCooldownRef.current = Date.now();

    const confNum = parseFloat(confStr);
    setStats(prev => ({
      count: prev.count + 1,
      totalConf: prev.totalConf + (isNaN(confNum) ? 0 : confNum),
      freq: { ...prev.freq, [sign]: (prev.freq[sign] || 0) + 1 },
    }));
  }, [dispatchSignWord]);

  const runGestureLoop = useCallback(() => {
    if (!isRunningRef.current) return;
    const video = localVidRef.current;
    const recognizer = gestureRef.current;

    const now = performance.now();
    const timeDelta = now - lastProcessingTimeRef.current; 

    if (
      recognizer && 
      video && 
      video.readyState >= 2 && 
      video.currentTime !== lastVideoTimeRef.current &&
      timeDelta >= 400 
    ) {
      lastVideoTimeRef.current = video.currentTime;
      lastProcessingTimeRef.current = now; 

      try {
        const MathResults = recognizer.recognizeForVideo(video, Date.now());
        if (MathResults.gestures && MathResults.gestures.length > 0) {
          let bestGesture = null;
          let bestScore   = 0;
          MathResults.gestures.forEach(handGestures => {
            if (handGestures.length > 0 && handGestures[0].score > bestScore) {
              bestScore   = handGestures[0].score;
              bestGesture = handGestures[0].categoryName;
            }
          });

          if (bestGesture) {
            const pretty = prettifyGesture(bestGesture);
            const confStr = bestScore.toFixed(2);

            gestureHistoryRef.current.push(pretty);
            if (gestureHistoryRef.current.length > 3) {
              gestureHistoryRef.current.shift();
            }

            const allMatch = gestureHistoryRef.current.every(g => g === pretty);

            if (allMatch) {
              setCurrentGesture(pretty);
              setConfidence(confStr);

              if (pretty !== lastGestureRef.current) {
                lastGestureRef.current = pretty;
                processLiveGesture(pretty, confStr);
              }
            }
            drawLandmarks(MathResults);
          } else {
            handleEmptyFrame();
          }
        } else {
          handleEmptyFrame();
        }
      } catch (err) {}
    }
    rafRef.current = requestAnimationFrame(runGestureLoop);
  }, [processLiveGesture]);

  const handleEmptyFrame = () => {
    gestureHistoryRef.current.push('None');
    if (gestureHistoryRef.current.length > 3) gestureHistoryRef.current.shift();
    
    if (gestureHistoryRef.current.every(g => g === 'None')) {
      setCurrentGesture('None');
      lastGestureRef.current = 'None';
    }
    clearCanvas();
  };

  const drawLandmarks = (results) => {
    const canvas = canvasRef.current;
    const video  = localVidRef.current;
    if (!canvas || !video) return;
    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!results.landmarks) return;

    const CONNECTIONS = [
      [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],
      [0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[14,16],
      [0,17],[17,18],[18,19],[19,20],[5,9],[9,13],[13,17],
    ];

    results.landmarks.forEach(landmarks => {
      ctx.strokeStyle = 'rgba(0, 212, 255, 0.8)';
      ctx.lineWidth   = 2;
      CONNECTIONS.forEach(([a, b]) => {
        ctx.beginPath();
        ctx.moveTo(landmarks[a].x * canvas.width, landmarks[a].y * canvas.height);
        ctx.lineTo(landmarks[b].x * canvas.width, landmarks[b].y * canvas.height);
        ctx.stroke();
      });
      landmarks.forEach((lm, i) => {
        ctx.beginPath();
        ctx.arc(lm.x * canvas.width, lm.y * canvas.height, i === 0 ? 6 : 4, 0, Math.PI * 2);
        ctx.fillStyle = i === 0 ? '#3a8ef6' : '#00e5a0';
        ctx.fill();
      });
    });
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    if (canvas) canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
  };

  const syncPeers = useCallback(() => {
    const list = [];
    peersRef.current.forEach(({ peer, name: n }, id) => {
      list.push({ peerId: id, peer, name: n });
    });
    setActivePeers([...list]);
  }, []);

  const makePeer = useCallback(({ remoteId, initiator, stream, incomingSignal }) => {
    const peer = new Peer({ initiator, trickle: true, stream, config: ICE_CONFIG });
    peer.on('signal', signal => {
      socketRef.current?.emit('signal', { to: remoteId, signal });
    });
    peer.on('stream',  () => syncPeers());
    peer.on('connect', () => syncPeers());
    peer.on('close',   () => { 
      peersRef.current.delete(remoteId); 
      if (selectedPeerId === remoteId) setSelectedPeerId(null);
      syncPeers(); 
    });
    peer.on('error',   err => console.warn('[peer error]', err.message));

    if (!initiator && incomingSignal) peer.signal(incomingSignal);
    return peer;
  }, [syncPeers, selectedPeerId]);

  const startCamera = async () => {
    if (isRunningRef.current) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
      streamRef.current = stream;

      stream.getAudioTracks().forEach(track => {
        track.enabled = !isMuted;
      });

      if (localVidRef.current) {
        localVidRef.current.srcObject = stream;
        await localVidRef.current.play().catch(() => {});
      }

      isRunningRef.current = true;
      setIsRunning(true);

      if (gestureRef.current) {
        rafRef.current = requestAnimationFrame(runGestureLoop);
      }

      const socket = io(SOCKET_URL, { transports: ['websocket'], reconnection: false });
      socketRef.current = socket;

      socket.on('connect', () => {
        socket.emit('joinRoom', { roomId: currentRoomId, userName });
      });

      socket.on('chatMessage', ({ msg, senderName }) => {
        const ts = new Date().toLocaleTimeString('en-US', { hour12: false });
        setHistoryFeed(prev => [
          { id: Date.now(), type: 'chat', icon: '💬', text: `${senderName}: ${msg}`, meta: ts },
          ...prev.slice(0, 49)
        ]);

        if (msg.startsWith('[Sign]')) {
          speakText(`${senderName} says: ${msg.replace('[Sign]', '')}`);
        } else {
          speakText(`${senderName} chats: ${msg}`);
        }
      });

      // 🌟 FIX: Full instant array synchronization prevents connection drops for 4+ users
      socket.on('allUsers', users => {
        users.forEach(user => {
          if (peersRef.current.has(user.id)) return;
          const peer = makePeer({ remoteId: user.id, initiator: true, stream: streamRef.current });
          peersRef.current.set(user.id, { peer, name: user.name });
        });
        syncPeers();
      });

      socket.on('userJoined', ({ id, name }) => {
        if (!peersRef.current.has(id)) {
          peersRef.current.set(id, { peer: null, name });
          syncPeers();
        }
      });

      socket.on('signal', ({ from, signal }) => {
        let record = peersRef.current.get(from);
        if (!record || !record.peer) {
          const peer = makePeer({
            remoteId: from,
            initiator: false,
            stream: streamRef.current,
            incomingSignal: signal,
          });
          peersRef.current.set(from, { peer, name: record?.name || 'Peer' });
          syncPeers();
        } else {
          try { record.peer.signal(signal); } catch (e) {}
        }
      });

      socket.on('userLeft', id => {
        const r = peersRef.current.get(id);
        if (r?.peer) r.peer.destroy();
        peersRef.current.delete(id);
        if (selectedPeerId === id) setSelectedPeerId(null);
        syncPeers();
      });

    } catch (err) {
      console.error('[startCamera]', err);
    }
  };

  const toggleMute = () => {
    const nextMuteState = !isMuted;
    setIsMuted(nextMuteState);
    if (streamRef.current) {
      streamRef.current.getAudioTracks().forEach(track => {
        track.enabled = !nextMuteState;
      });
    }
  };

  const stopCamera = useCallback(() => {
    isRunningRef.current = false;
    setIsRunning(false);
    setCurrentGesture('None');
    setConfidence('—');
    lastGestureRef.current = 'None';
    setIsMuted(false); 
    setSelectedPeerId(null);
    cancelAnimationFrame(rafRef.current);
    clearCanvas();

    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
    if (localVidRef.current) localVidRef.current.srcObject = null;

    peersRef.current.forEach(({ peer }) => peer?.destroy());
    peersRef.current.clear();
    setActivePeers([]);

    socketRef.current?.disconnect();
    socketRef.current = null;
  }, [selectedPeerId]);

  const copyInviteLink = () => {
    navigator.clipboard.writeText(`${window.location.origin}/workspace?room=${currentRoomId}`)
      .then(() => { setLinkCopied(true); setTimeout(() => setLinkCopied(false), 2500); });
  };

  const sendChatMessage = (e) => {
    if (e.key && e.key !== 'Enter') return;
    if (!chatMessage.trim()) return;

    const ts = new Date().toLocaleTimeString('en-US', { hour12: false });
    
    if (socketRef.current) {
      socketRef.current.emit('chatMessage', { roomId: currentRoomId, msg: chatMessage, senderName: userName });
    }

    setHistoryFeed(prev => [
      { id: Date.now(), type: 'chat', icon: '💬', text: `You: ${chatMessage}`, meta: ts },
      ...prev.slice(0, 49)
    ]);

    speakText(`You said: ${chatMessage}`);
    setChatMessage('');
  };

  const handleLogout = () => { stopCamera(); localStorage.clear(); navigate('/login'); };

  const getTopGesture = () => {
    let top = '—', max = 0;
    Object.entries(stats.freq).forEach(([k, v]) => { if (v > max) { max = v; top = k; } });
    return top.split(' ')[0];
  };

  useEffect(() => {
    if (modelStatus === 'ready' && isRunningRef.current && !rafRef.current) {
      rafRef.current = requestAnimationFrame(runGestureLoop);
    }
  }, [modelStatus, runGestureLoop]);

  useEffect(() => () => {
    stopCamera();
    gestureRef.current?.close();
  }, []); // eslint-disable-line

  const modelBadge = {
    idle:    { color: '#7a99cc', text: 'Model: idle' },
    loading: { color: '#f5a623', text: 'Model: loading…' },
    ready:   { color: '#00e5a0', text: 'Model: ready ✓' },
    error:   { color: '#e84040', text: 'Model: load failed ✗' },
  }[modelStatus];

  return (
    <div className="app-shell">
      {showWelcomeModal && (
        <div className="modal-overlay">
          <div className="modal-content">
            <div className="modal-header">
              <h2>Welcome to the Sign Workspace, {userName}! 👋</h2>
              <div className="underline"></div>
            </div>
            <div className="modal-body">
              <p>Your collaborative workspace room is set up. Keep these live operational tips in mind before initializing:</p>
              <ul className="modal-guidelines">
                <li>
                  <span className="bullet-icon">🔊</span> 
                  <strong>Text-to-Speech Engine:</strong> Integrated browser speech utilities will automatically speak all translated messages out loud!
                </li>
                <li>
                  <span className="bullet-icon">🖥️</span> 
                  <strong>Theater Sizing:</strong> Click on any user's bottom profile layout card to magnify their video feed onto the large primary projection grid panel.
                </li>
                <li>
                  <span className="bullet-icon">⏹️</span> 
                  <strong>Broadcasting Text:</strong> Flash the <strong>"Space"</strong> hand configuration to immediately pack your buffer and relay the translated text across the room.
                </li>
              </ul>
            </div>
            <div className="modal-footer">
              <button className="modal-start-btn" onClick={() => setShowWelcomeModal(false)}>
                Enter Workspace & Start Room
              </button>
            </div>
          </div>
        </div>
      )}

      <section className="video-section">
        <div className="video-header">
          <div className="header-left">
            <div className="rec-dot" />
            <span className="rec-label">LIVE ROOM</span>
            <span className="session-id">{currentRoomId}</span>
            <button onClick={copyInviteLink} style={{
              marginLeft: '12px', padding: '4px 10px', borderRadius: '4px',
              fontSize: '0.7rem', cursor: 'pointer',
              backgroundColor: linkCopied ? '#2e7d32' : '#1976d2',
              color: '#fff', border: 'none', fontWeight: 'bold', transition: 'all 0.2s',
            }}>
              {linkCopied ? '✓ Copied!' : '🔗 Copy Invite Link'}
            </button>
            <span style={{
              marginLeft: '12px', fontSize: '0.65rem', fontFamily: 'var(--font-mono)',
              color: modelBadge.color, fontWeight: 600,
            }}>
              {modelBadge.text}
            </span>
          </div>
          <div className="header-title">Sign Language Interpreter Module</div>
        </div>

        {/* Primary View Container (Changes dynamically based on user selection) */}
        <div className="main-feed-wrap">
          <video 
            ref={theaterVidRef} 
            className="theater-view-element" 
            style={{ display: selectedPeerId ? 'block' : 'none' }} 
            autoPlay 
            playsInline 
          />
          
          <video 
            ref={localVidRef} 
            id="webcam" 
            autoPlay 
            playsInline 
            muted
            style={{ display: (!selectedPeerId && isRunning) ? 'block' : 'none' }} 
          />
          
          <canvas 
            ref={canvasRef} 
            id="overlay"
            style={{
              display: !selectedPeerId && isRunning ? 'block' : 'none',
              position: 'absolute', inset: 0,
              width: '100%', height: '100%',
              pointerEvents: 'none',
              transform: 'scaleX(-1)',   
            }}
          />
          
          {(!isRunning && !selectedPeerId) && (
            <div className="feed-placeholder">
              <div className="placeholder-text">
                {modelStatus === 'loading'
                  ? 'LOADING GESTURE MODEL…'
                  : modelStatus === 'error'
                  ? 'MODEL LOAD FAILED — CHECK CONSOLE'
                  : 'CAMERA INACTIVE · PRESS ▶️ TO START'}
              </div>
            </div>
          )}

          <div className="gesture-badge">
            <div className="gesture-icon">
              {GESTURE_ICONS[currentGesture.replace(/ /g, '_')] || GESTURE_ICONS[currentGesture] || '🤚'}
            </div>
            <div className="gesture-info">
              <span className="gesture-label">DETECTED SIGN</span>
              <span className="gesture-value">{currentGesture}</span>
            </div>
            <div className="gesture-conf">
              <span className="conf-label">BUFFER WORD</span>
              <span className="conf-value" style={{ color: 'var(--accent-cyan)' }}>
                {accumulatedWord || '—'}
              </span>
            </div>
          </div>
        </div>

        {/* Multi-User Flex Rows Grid */}
        <div className="participants-row">
          <div 
            className={`participant-card ${!selectedPeerId ? 'selected-focus-border' : 'active-speaker'}`}
            onClick={() => setSelectedPeerId(null)}
            style={{ cursor: 'pointer' }}
          >
            <div className="remote-video-wrap local-avatar-wrap">
              {selectedPeerId && isRunning ? (
                <video 
                  srcObject={streamRef.current} 
                  autoPlay 
                  playsInline 
                  muted 
                  style={{ width: '100%', height: '100%', objectFit: 'cover', transform: 'scaleX(-1)' }}
                  ref={(el) => { if (el && streamRef.current) el.srcObject = streamRef.current; }}
                />
              ) : (
                <div className="participant-avatar">ME</div>
              )}
            </div>
            <span className="participant-name">{userName} (You)</span>
            <div className="participant-status active" />
          </div>
          
          {activePeers.map(p => (
            <VideoNode 
              key={p.peerId} 
              peer={p.peer} 
              name={p.name} 
              isSelected={selectedPeerId === p.peerId}
              onClick={() => setSelectedPeerId(p.peerId)}
            />
          ))}
        </div>

        <div className="control-bar">
          <button className="ctrl-btn" onClick={startCamera}
            disabled={isRunning || modelStatus === 'loading'}
            title="Start Camera"
            style={{ color: isRunning ? 'var(--accent-green)' : 'inherit' }}>▶️</button>
          
          <button className="ctrl-btn" onClick={toggleMute} disabled={!isRunning}
            style={{ color: isMuted ? 'var(--accent-red)' : 'inherit' }}
            title="Mute Mic">
            {isMuted ? '🔇' : '🎙️'}
          </button>
          
          <button className="ctrl-btn" onClick={stopCamera} disabled={!isRunning} title="Pause Room">⏸️</button>
          <button className="ctrl-btn end-call" onClick={handleLogout} title="Leave Session">❌</button>
        </div>
      </section>

      <aside className="sidebar">
        <div className="sidebar-header">
          <span className="sidebar-title">Translation & Chat Feed</span>
        </div>
        
        <div style={{
          padding: '12px 14px',
          borderBottom: '1px solid var(--navy-border)',
          background: 'rgba(58, 142, 246, 0.04)'
        }}>
          <span className="feed-heading" style={{ color: 'var(--accent-cyan)', marginBottom: '6px', display: 'block' }}>
            System Guide
          </span>
          <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', lineHeight: '1.4', display: 'flex', flexDirection: 'column', gap: '4px' }}>
            <div>• <strong>Theater View:</strong> Click any user's panel to enlarge their feed.</div>
            <div>• <strong>Stabilized Window:</strong> Signs verify across consecutive frames.</div>
            <div>• <strong>Space Trigger:</strong> Show "Space" sign to submit string to chat.</div>
          </div>
        </div>

        <div className="sign-feed">
          <span className="feed-heading">Activity Log</span>
          <ul className="feed-list">
            {historyFeed.length === 0
              ? <li className="feed-item placeholder"><div className="feed-text"><span className="feed-sign">Awaiting activity data…</span></div></li>
              : historyFeed.map(item => (
                <li key={item.id} className="feed-item">
                  <div className="feed-avatar" style={item.type === 'chat'
                    ? { background: 'linear-gradient(135deg, var(--accent-green), var(--navy-panel))' } : {}}>
                    {item.icon}
                  </div>
                  <div className="feed-text">
                    <span className="feed-sign">{item.text}</span>
                    <span className="feed-time">{item.meta}</span>
                  </div>
                </li>
              ))
            }
          </ul>
        </div>
        <div className="stats-strip">
          <div className="stat"><span className="stat-val">{stats.count}</span><span className="stat-key">Signs</span></div>
          <div className="stat">
            <span className="stat-val">{stats.count > 0 ? (stats.totalConf / stats.count).toFixed(2) : '—'}</span>
            <span className="stat-key">Avg Conf</span>
          </div>
          <div className="stat"><span className="stat-val">{getTopGesture()}</span><span className="stat-key">Top Sign</span></div>
        </div>
        <div className="chat-input-wrap">
          <input type="text" className="chat-input" placeholder="Type note or chat text…"
            value={chatMessage} onChange={e => setChatMessage(e.target.value)} onKeyDown={sendChatMessage} />
          <button className="send-btn" onClick={() => sendChatMessage({ key: 'Enter' })}>➡️</button>
        </div>
      </aside>
    </div>
  );
};

export default VideoCall;
