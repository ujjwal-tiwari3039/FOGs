'use client'

import { useState, useEffect } from 'react'

const CheckCircle = () => <span className="text-green-500 mr-2">🟢</span>
const XCircle = () => <span className="text-red-500 mr-2">🔴</span>
const AlertCircle = () => <span className="text-amber-500 mr-2">🟠</span>

export default function Home() {
  const [showIntro, setShowIntro] = useState(true)
  const [tab, setTab] = useState<'single' | 'batch'>('single')
  
  // Single Answer State
  const [question, setQuestion] = useState('Explain why you got a voltage reading of 1.5...')
  const [refs, setRefs] = useState(['Terminal 1 and the positive terminal are separated by the gap'])
  const [student, setStudent] = useState('Because there is a gap')
  const [result, setResult] = useState<any>(null)
  const [isGrading, setIsGrading] = useState(false)
  
  // Batch State
  const [refText, setRefText] = useState('1. What is X?\nIt is Y.')
  const [refPairs, setRefPairs] = useState<any[]>([])
  const [stuText, setStuText] = useState('1. What is X?\nI think it is Y.')
  const [stuPairs, setStuPairs] = useState<any[]>([])
  const [batchResults, setBatchResults] = useState<any[]>([])
  const [isBatching, setIsBatching] = useState(false)

  useEffect(() => {
    // Hide intro after 2 seconds
    const timer = setTimeout(() => setShowIntro(false), 2000)
    return () => clearTimeout(timer)
  }, [])

  const handleGradeSingle = async () => {
    setIsGrading(true)
    try {
      const res = await fetch('/api/grade_single', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: question,
          references: refs.filter(r => r.trim()),
          student_answer: student
        })
      })
      setResult(await res.json())
    } catch (e) {
      console.error(e)
    } finally {
      setIsGrading(false)
    }
  }
  
  const handleExtractRefs = async () => {
    const formData = new FormData()
    formData.append('text', refText)
    const res = await fetch('/api/extract_text', {
      method: 'POST', body: formData
    })
    setRefPairs(await res.json())
  }
  
  const handleExtractStu = async () => {
    const formData = new FormData()
    formData.append('text', stuText)
    const res = await fetch('/api/extract_text', {
      method: 'POST', body: formData
    })
    setStuPairs(await res.json())
  }
  
  const handleBatchGrade = async () => {
    setIsBatching(true)
    try {
      const resArr = []
      const refMap: Record<number, string> = {}
      refPairs.forEach(r => refMap[r.number] = r.answer_text)
      
      for (const stu of stuPairs) {
        if (refMap[stu.number]) {
          const res = await fetch('/api/grade_single', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              question: stu.question_text || `Question ${stu.number}`,
              references: [refMap[stu.number]],
              student_answer: stu.answer_text
            })
          })
          const data = await res.json()
          resArr.push({ ...data, number: stu.number, stu_ans: stu.answer_text, ref: refMap[stu.number] })
        }
      }
      resArr.sort((a, b) => (a.needs_review === b.needs_review ? 0 : a.needs_review ? -1 : 1))
      setBatchResults(resArr)
    } finally {
      setIsBatching(false)
    }
  }

  return (
    <main className="min-h-screen bg-[#FCF9F3] text-[#1D1D1B] font-sans relative overflow-hidden">
      
      {/* Intro Animation Overlay */}
      <div 
        className={`fixed inset-0 z-50 bg-[#0A1017] flex flex-col items-center justify-center transition-transform duration-1000 ease-[cubic-bezier(0.85,0,0.15,1)] ${showIntro ? 'translate-y-0' : '-translate-y-full'}`}
      >
        <img src="/logo.jpeg" alt="FOGs Logo" className="w-24 h-24 rounded-full mb-6 shadow-lg object-cover" />
        <h1 className="text-white text-3xl md:text-5xl font-serif tracking-widest uppercase" style={{ fontFamily: 'var(--font-playfair)' }}>
          FOGs <span className="text-[#FFC72C]">GRADER</span>
        </h1>
      </div>

      {/* Top Navbar */}
      <nav className="w-full py-6 px-8 flex justify-between items-center">
        <div className="flex items-center gap-3 font-serif font-black text-2xl tracking-tighter" style={{ fontFamily: 'var(--font-playfair)' }}>
          <img src="/logo.jpeg" alt="FOGs Logo" className="w-10 h-10 rounded-full object-cover shadow-sm" />
          <span>FOGs</span>
        </div>
        <div className="flex space-x-2 bg-white rounded-full p-1 shadow-sm border border-gray-100">
          <button 
            className={`px-6 py-2 rounded-full text-sm font-semibold transition-all ${tab === 'single' ? 'bg-[#FFC72C] text-black shadow-sm' : 'bg-transparent text-gray-500 hover:text-black'}`}
            onClick={() => setTab('single')}
          >
            Single Answer
          </button>
          <button 
            className={`px-6 py-2 rounded-full text-sm font-semibold transition-all ${tab === 'batch' ? 'bg-[#FFC72C] text-black shadow-sm' : 'bg-transparent text-gray-500 hover:text-black'}`}
            onClick={() => setTab('batch')}
          >
            Batch Grade
          </button>
        </div>
        <button className="bg-[#1D1D1B] text-white px-5 py-2 rounded-full text-sm font-medium hover:bg-black transition-colors">
          Sign In ↗
        </button>
      </nav>

      {/* Hero Section */}
      <div className="max-w-4xl mx-auto mt-8 mb-12 px-6 text-center">
        <h1 className="text-5xl md:text-7xl font-black text-[#1D1D1B] mb-4 tracking-tight leading-tight" style={{ fontFamily: 'var(--font-playfair)' }}>
          DISCOVER <span className="text-[#FFC72C]">INSIGHTS.</span><br />
          GRADE WITH CONFIDENCE.
        </h1>
        <p className="text-gray-500 text-lg md:text-xl max-w-2xl mx-auto">
          The ultimate semantic assessment platform. Grade short answers instantly, track 
          robustness against counterfactuals, and route edge cases without the friction.
        </p>
      </div>

      <div className="max-w-5xl mx-auto px-6 pb-24">
        
        {/* ================= SINGLE ANSWER TAB ================= */}
        {tab === 'single' && (
          <div className="bg-white rounded-[2rem] shadow-[0_8px_30px_rgb(0,0,0,0.04)] overflow-hidden border border-gray-100">
            <div className="h-4 bg-[#EBEBFF] w-full"></div>
            <div className="p-8 md:p-12 space-y-8">
              
              <div>
                <label className="block text-sm font-bold text-gray-400 uppercase tracking-wider mb-2">The Question</label>
                <input 
                  className="w-full text-xl font-medium border-b-2 border-gray-100 focus:border-[#FFC72C] outline-none py-2 bg-transparent transition-colors" 
                  value={question} 
                  onChange={e => setQuestion(e.target.value)} 
                />
              </div>
              
              <div>
                <label className="block text-sm font-bold text-gray-400 uppercase tracking-wider mb-2">Reference Answers</label>
                <div className="space-y-3">
                  {refs.map((r, i) => (
                    <input 
                      key={i} 
                      className="w-full text-lg border bg-gray-50 border-gray-100 rounded-xl px-4 py-3 focus:bg-white focus:border-[#FFC72C] outline-none transition-colors" 
                      value={r} 
                      placeholder="Type reference answer..."
                      onChange={e => {
                        const newRefs = [...refs]
                        newRefs[i] = e.target.value
                        setRefs(newRefs)
                      }} 
                    />
                  ))}
                </div>
                <button 
                  className="mt-3 text-[#1D1D1B] font-semibold text-sm hover:text-[#FFC72C] transition-colors" 
                  onClick={() => setRefs([...refs, ''])}
                >
                  + Add Alternative Reference
                </button>
              </div>
              
              <div>
                <label className="block text-sm font-bold text-gray-400 uppercase tracking-wider mb-2">Student's Submission</label>
                <textarea 
                  className="w-full text-lg border bg-gray-50 border-gray-100 rounded-xl px-4 py-3 focus:bg-white focus:border-[#FFC72C] outline-none transition-colors min-h-[120px] resize-y" 
                  value={student} 
                  onChange={e => setStudent(e.target.value)} 
                />
              </div>
              
              <div className="pt-4 flex justify-between items-center border-t border-gray-100">
                <div className="text-gray-400 text-sm font-medium">✨ Powered by Qwen2.5-3B-Instruct</div>
                <button 
                  className="bg-[#FFC72C] text-[#1D1D1B] px-8 py-3 rounded-full font-bold text-lg hover:scale-105 transition-transform disabled:opacity-50 shadow-sm" 
                  onClick={handleGradeSingle}
                  disabled={isGrading}
                >
                  {isGrading ? 'Grading...' : 'Grade Answer ↗'}
                </button>
              </div>

              {result && (
                <div className="mt-8 p-6 bg-[#FCF9F3] border border-gray-200 rounded-2xl animate-in fade-in slide-in-from-bottom-4 duration-500">
                  <div className="flex items-center text-2xl font-bold mb-4 capitalize font-serif" style={{ fontFamily: 'var(--font-playfair)' }}>
                    {result.label === 'correct' ? <CheckCircle/> : result.label === 'contradictory' ? <AlertCircle/> : <XCircle/>}
                    {result.label}
                  </div>
                  
                  <div className={`p-4 rounded-xl mb-4 font-semibold flex items-center justify-between ${result.needs_review ? 'bg-amber-100 text-amber-900' : 'bg-green-100 text-green-900'}`}>
                    <span>Confidence Score</span>
                    <span className="text-lg">{(result.confidence * 100).toFixed(1)}%</span>
                  </div>
                  {result.needs_review && (
                    <div className="mb-4 text-amber-700 text-sm font-bold bg-amber-50 py-2 px-3 rounded-lg inline-block border border-amber-200">
                      ⚠️ Flagged for Human Review
                    </div>
                  )}
                  
                  <div className="space-y-2">
                    <p className="text-gray-700 leading-relaxed"><strong className="text-black">Evidence:</strong> {result.evidence}</p>
                    {result.note && <p className="text-sm text-gray-500 bg-white p-3 rounded-lg border">ℹ️ Note: {result.note}</p>}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
        
        {/* ================= BATCH GRADING TAB ================= */}
        {tab === 'batch' && (
          <div className="space-y-8 animate-in fade-in duration-500">
            <div className="grid md:grid-cols-2 gap-8">
              
              {/* References Column */}
              <div className="bg-white rounded-[2rem] shadow-[0_8px_30px_rgb(0,0,0,0.04)] overflow-hidden border border-gray-100 flex flex-col">
                <div className="h-4 bg-[#FFE8E8] w-full"></div>
                <div className="p-8 flex-1">
                  <div className="flex justify-between items-center mb-6">
                    <h2 className="text-2xl font-bold font-serif" style={{ fontFamily: 'var(--font-playfair)' }}>References</h2>
                    <span className="text-xs font-bold bg-gray-100 px-3 py-1 rounded-full uppercase text-gray-500">Step 1</span>
                  </div>
                  
                  <textarea 
                    className="w-full text-sm border bg-gray-50 border-gray-100 rounded-xl px-4 py-3 focus:bg-white focus:border-[#FFC72C] outline-none transition-colors mb-4" 
                    rows={6} 
                    value={refText} 
                    onChange={e => setRefText(e.target.value)} 
                    placeholder="1. Question text&#10;Answer text"
                  />
                  <button className="bg-gray-100 hover:bg-gray-200 text-black px-5 py-2 rounded-full text-sm font-semibold transition-colors mb-6" onClick={handleExtractRefs}>
                    Extract Pairs
                  </button>
                  
                  {refPairs.length > 0 && (
                    <div className="space-y-3 max-h-[300px] overflow-y-auto pr-2">
                      <h3 className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-2">Confirm Extraction</h3>
                      {refPairs.map(p => (
                        <div key={p.number} className="flex gap-3 items-start bg-[#FCF9F3] p-3 rounded-xl border border-gray-100">
                          <div className="w-8 h-8 rounded-full bg-white shadow-sm flex items-center justify-center font-bold text-sm shrink-0">{p.number}</div>
                          <textarea 
                            className="flex-1 bg-transparent border-none outline-none text-sm resize-none" 
                            value={p.answer_text} 
                            rows={2}
                            onChange={e => {
                              setRefPairs(refPairs.map(x => x.number === p.number ? {...x, answer_text: e.target.value} : x))
                            }}
                          />
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>

              {/* Student Column */}
              <div className="bg-white rounded-[2rem] shadow-[0_8px_30px_rgb(0,0,0,0.04)] overflow-hidden border border-gray-100 flex flex-col">
                <div className="h-4 bg-[#E2F5EA] w-full"></div>
                <div className="p-8 flex-1">
                  <div className="flex justify-between items-center mb-6">
                    <h2 className="text-2xl font-bold font-serif" style={{ fontFamily: 'var(--font-playfair)' }}>Student Answers</h2>
                    <span className="text-xs font-bold bg-gray-100 px-3 py-1 rounded-full uppercase text-gray-500">Step 2</span>
                  </div>
                  
                  <textarea 
                    className="w-full text-sm border bg-gray-50 border-gray-100 rounded-xl px-4 py-3 focus:bg-white focus:border-[#FFC72C] outline-none transition-colors mb-4" 
                    rows={6} 
                    value={stuText} 
                    onChange={e => setStuText(e.target.value)} 
                  />
                  <button className="bg-gray-100 hover:bg-gray-200 text-black px-5 py-2 rounded-full text-sm font-semibold transition-colors mb-6" onClick={handleExtractStu}>
                    Extract Answers
                  </button>
                  
                  {stuPairs.length > 0 && (
                    <div className="space-y-3 max-h-[300px] overflow-y-auto pr-2">
                      <h3 className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-2">Confirm OCR</h3>
                      {stuPairs.map(p => (
                        <div key={p.number} className="flex gap-3 items-start bg-[#FCF9F3] p-3 rounded-xl border border-gray-100">
                          <div className="w-8 h-8 rounded-full bg-white shadow-sm flex items-center justify-center font-bold text-sm shrink-0">{p.number}</div>
                          <textarea 
                            className="flex-1 bg-transparent border-none outline-none text-sm resize-none" 
                            value={p.answer_text} 
                            rows={2}
                            onChange={e => {
                              setStuPairs(stuPairs.map(x => x.number === p.number ? {...x, answer_text: e.target.value} : x))
                            }}
                          />
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>

            </div>
            
            <div className="flex justify-center mt-8">
              <button 
                className="bg-[#1D1D1B] text-white px-10 py-4 rounded-full font-bold text-lg hover:bg-black transition-all hover:shadow-lg disabled:opacity-50" 
                onClick={handleBatchGrade}
                disabled={isBatching}
              >
                {isBatching ? 'Grading Batch...' : 'Grade All Answers ↗'}
              </button>
            </div>
            
            {batchResults.length > 0 && (
              <div className="mt-12 space-y-4">
                <h3 className="text-2xl font-bold font-serif mb-6" style={{ fontFamily: 'var(--font-playfair)' }}>Grading Results</h3>
                {batchResults.map((r, i) => (
                  <details key={i} className="group bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden">
                    <summary className="font-semibold cursor-pointer p-5 flex items-center justify-between hover:bg-gray-50 transition-colors list-none">
                      <div className="flex items-center gap-6">
                        <span className="w-10 h-10 rounded-full bg-[#FCF9F3] flex items-center justify-center font-bold text-sm">{r.number}</span>
                        <div className="flex flex-col">
                          <span className="text-xs text-gray-400 uppercase tracking-wider">Prediction</span>
                          <span className="capitalize text-lg">{r.label === 'correct' ? '🟢 Correct' : r.label === 'contradictory' ? '🟠 Contradictory' : '🔴 Incorrect'}</span>
                        </div>
                        <div className="hidden md:flex flex-col ml-8 border-l pl-8 border-gray-100">
                          <span className="text-xs text-gray-400 uppercase tracking-wider">Status</span>
                          <span className={`${r.needs_review ? 'text-amber-600 font-bold' : 'text-green-600'}`}>
                            {r.needs_review ? '⚠️ Review Required' : '✓ Auto-Graded'}
                          </span>
                        </div>
                      </div>
                      <div className="text-gray-400 group-open:rotate-180 transition-transform">▼</div>
                    </summary>
                    <div className="p-6 pt-0 border-t border-gray-100 bg-[#FCF9F3] mt-2 text-sm">
                      <div className="grid md:grid-cols-2 gap-6 mt-4">
                        <div className="bg-white p-4 rounded-xl border border-gray-100">
                          <h4 className="text-xs font-bold text-gray-400 uppercase mb-2">Student Answer</h4>
                          <p className="text-gray-800">{r.stu_ans}</p>
                        </div>
                        <div className="bg-white p-4 rounded-xl border border-gray-100">
                          <h4 className="text-xs font-bold text-gray-400 uppercase mb-2">Reference Answer</h4>
                          <p className="text-gray-800">{r.ref}</p>
                        </div>
                      </div>
                      <div className="mt-4 bg-white p-4 rounded-xl border border-gray-100">
                        <h4 className="text-xs font-bold text-gray-400 uppercase mb-2">AI Evidence</h4>
                        <p className="text-gray-800">{r.evidence}</p>
                        {r.note && <p className="mt-2 text-amber-700 bg-amber-50 p-2 rounded text-xs">{r.note}</p>}
                      </div>
                    </div>
                  </details>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </main>
  )
}
