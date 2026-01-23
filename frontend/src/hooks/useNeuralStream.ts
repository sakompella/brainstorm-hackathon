import { useEffect, useRef, useState, useCallback } from 'react'

export interface InitMessage {
  type: 'init'
  channels_coords: [number, number][]
  grid_size: number
  fs: number
  batch_size: number
}

export interface SampleBatchMessage {
  type: 'sample_batch'
  neural_data: number[][]
  start_time_s: number
  sample_count: number
  fs: number
}

export type StreamMessage = InitMessage | SampleBatchMessage

export interface NeuralStreamState {
  isConnected: boolean
  isConnecting: boolean
  channelCoords: [number, number][] | null
  gridSize: number
  latestData: number[] | null
  time: number
  fps: number
}

export function useNeuralStream(url: string = 'ws://localhost:8765') {
  const wsRef = useRef<WebSocket | null>(null)
  const frameCountRef = useRef(0)
  const lastFpsUpdateRef = useRef(performance.now())

  const [state, setState] = useState<NeuralStreamState>({
    isConnected: false,
    isConnecting: false,
    channelCoords: null,
    gridSize: 32,
    latestData: null,
    time: 0,
    fps: 0,
  })

  // Store latest sample for rendering (no averaging, lowest latency)
  const latestDataRef = useRef<number[] | null>(null)
  const latestTimeRef = useRef(0)

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.close()
      return
    }

    setState(s => ({ ...s, isConnecting: true }))

    const ws = new WebSocket(url)
    wsRef.current = ws

    ws.onopen = () => {
      setState(s => ({ ...s, isConnected: true, isConnecting: false }))
    }

    ws.onmessage = (event) => {
      try {
        const data: StreamMessage = JSON.parse(event.data)

        if (data.type === 'init') {
          setState(s => ({
            ...s,
            channelCoords: data.channels_coords,
            gridSize: data.grid_size,
          }))
        } else if (data.type === 'sample_batch') {
          // Use the last sample in the batch for lowest latency
          const lastSample = data.neural_data[data.neural_data.length - 1]
          const lastTime = data.start_time_s + (data.sample_count - 1) / data.fs

          latestDataRef.current = lastSample
          latestTimeRef.current = lastTime

          // Update FPS counter
          frameCountRef.current++
          const now = performance.now()
          if (now - lastFpsUpdateRef.current >= 1000) {
            setState(s => ({
              ...s,
              latestData: latestDataRef.current,
              time: latestTimeRef.current,
              fps: frameCountRef.current,
            }))
            frameCountRef.current = 0
            lastFpsUpdateRef.current = now
          } else {
            setState(s => ({
              ...s,
              latestData: latestDataRef.current,
              time: latestTimeRef.current,
            }))
          }
        }
      } catch (err) {
        console.error('Error parsing message:', err)
      }
    }

    ws.onerror = () => {
      setState(s => ({ ...s, isConnected: false, isConnecting: false }))
    }

    ws.onclose = () => {
      setState(s => ({ ...s, isConnected: false, isConnecting: false }))
      wsRef.current = null
    }
  }, [url])

  const disconnect = useCallback(() => {
    wsRef.current?.close()
  }, [])

  useEffect(() => {
    return () => {
      wsRef.current?.close()
    }
  }, [])

  return {
    ...state,
    connect,
    disconnect,
  }
}
