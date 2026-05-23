"""
app.py
──────
Graphical user interface for the ASL fingerspelling recognition system.

Integrates the InferenceEngine with a Tkinter desktop application that
displays predicted letters, accumulates words, and provides session controls.

Usage
-----
    python app.py
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import time
import os
from datetime import datetime

from inference import InferenceEngine, PredictionResult


# ── Colour palette ────────────────────────────────────────────────────────────

COLOUR = {
    "bg"           : "#1E1E2E",   # dark background
    "panel"        : "#2A2A3E",   # slightly lighter panel
    "border"       : "#3A3A5E",   # subtle border
    "accent"       : "#7C9EF8",   # soft blue accent
    "accent_dark"  : "#4A6CF7",   # deeper blue
    "text"         : "#E0E0F0",   # primary text
    "text_dim"     : "#8888AA",   # secondary / dim text
    "green"        : "#4CAF82",   # high confidence
    "amber"        : "#F5A623",   # medium confidence
    "grey"         : "#555570",   # no detection
    "red"          : "#E05C5C",   # error / disconnected
    "white"        : "#FFFFFF",
}

FONT_FAMILY = "Arial"


# ── Main application ──────────────────────────────────────────────────────────

class ASLApp(tk.Tk):
    """
    Root Tkinter window.  Owns the InferenceEngine and orchestrates
    all child panels.
    """

    def __init__(self):
        super().__init__()

        self.title("ASL Fingerspelling Recognition")
        self.geometry("700x820")
        self.minsize(600, 720)
        self.configure(bg=COLOUR["bg"])
        self.resizable(True, True)

        # ── Engine ────────────────────────────────────────────────────────
        self.engine = InferenceEngine()
        self._running = False

        # ── State ─────────────────────────────────────────────────────────
        self._word_buffer   : list  = []   # letters accumulated so far
        self._word_history  : list  = []   # completed words
        self._session_log   : list  = []   # timestamped letter events
        self._debounce_buf  : list  = []   # last N letters for word-append guard
        self._last_appended : str   = ""   # letter most recently added to buffer

        # ── Build UI ──────────────────────────────────────────────────────
        self._build_ui()

        # ── Poll loop (every 50ms via Tkinter scheduler) ──────────────────
        self._poll()

        # ── Graceful shutdown ─────────────────────────────────────────────
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # Outer padding frame
        root_frame = tk.Frame(self, bg=COLOUR["bg"], padx=16, pady=12)
        root_frame.pack(fill=tk.BOTH, expand=True)

        self._build_header(root_frame)
        self._build_letter_display(root_frame)
        self._build_word_buffer(root_frame)
        self._build_history(root_frame)
        self._build_controls(root_frame)

    # ── Header ────────────────────────────────────────────────────────────────

    def _build_header(self, parent):
        frame = tk.Frame(parent, bg=COLOUR["bg"])
        frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            frame,
            text="ASL Fingerspelling Recognition",
            font=(FONT_FAMILY, 15, "bold"),
            bg=COLOUR["bg"],
            fg=COLOUR["accent"],
        ).pack(side=tk.LEFT)

        # Connection status pill
        self._status_var = tk.StringVar(value="● Stopped")
        self._status_label = tk.Label(
            frame,
            textvariable=self._status_var,
            font=(FONT_FAMILY, 10),
            bg=COLOUR["bg"],
            fg=COLOUR["text_dim"],
        )
        self._status_label.pack(side=tk.RIGHT, padx=4)

    # ── Letter display ────────────────────────────────────────────────────────

    def _build_letter_display(self, parent):
        panel = tk.Frame(
            parent,
            bg=COLOUR["panel"],
            highlightbackground=COLOUR["border"],
            highlightthickness=1,
        )
        panel.pack(fill=tk.X, pady=(0, 10), ipady=18)

        # Big letter
        self._letter_var = tk.StringVar(value="—")
        self._letter_label = tk.Label(
            panel,
            textvariable=self._letter_var,
            font=(FONT_FAMILY, 110, "bold"),
            bg=COLOUR["panel"],
            fg=COLOUR["grey"],
            width=3,
            anchor="center",
        )
        self._letter_label.pack()

        # Confidence text
        self._conf_var = tk.StringVar(value="No gesture detected")
        tk.Label(
            panel,
            textvariable=self._conf_var,
            font=(FONT_FAMILY, 11),
            bg=COLOUR["panel"],
            fg=COLOUR["text_dim"],
        ).pack(pady=(0, 6))

        # Confidence progress bar
        style = ttk.Style()
        style.theme_use("default")
        style.configure(
            "Conf.Horizontal.TProgressbar",
            troughcolor=COLOUR["border"],
            background=COLOUR["green"],
            thickness=10,
        )
        self._conf_val = tk.DoubleVar(value=0)
        ttk.Progressbar(
            panel,
            variable=self._conf_val,
            maximum=100,
            length=320,
            style="Conf.Horizontal.TProgressbar",
        ).pack(pady=(0, 10))

    # ── Word buffer ───────────────────────────────────────────────────────────

    def _build_word_buffer(self, parent):
        panel = tk.Frame(
            parent,
            bg=COLOUR["panel"],
            highlightbackground=COLOUR["border"],
            highlightthickness=1,
        )
        panel.pack(fill=tk.X, pady=(0, 10), ipady=10)

        tk.Label(
            panel,
            text="CURRENT WORD",
            font=(FONT_FAMILY, 9, "bold"),
            bg=COLOUR["panel"],
            fg=COLOUR["text_dim"],
        ).pack(anchor="w", padx=14, pady=(8, 2))

        # Word display
        self._word_var = tk.StringVar(value="")
        tk.Label(
            panel,
            textvariable=self._word_var,
            font=(FONT_FAMILY, 32, "bold"),
            bg=COLOUR["panel"],
            fg=COLOUR["white"],
            anchor="w",
            height=2,
        ).pack(fill=tk.X, padx=14)

        # Word control buttons
        btn_frame = tk.Frame(panel, bg=COLOUR["panel"])
        btn_frame.pack(fill=tk.X, padx=14, pady=(6, 2))

        for label, cmd, width in [
            ("⎵  Space",     self._on_space,     10),
            ("⌫  Backspace", self._on_backspace,  10),
            ("✕  Clear",     self._on_clear,       8),
        ]:
            _btn(btn_frame, label, cmd, width=width).pack(side=tk.LEFT, padx=(0, 6))

        # Completed words output
        tk.Label(
            panel,
            text="OUTPUT",
            font=(FONT_FAMILY, 9, "bold"),
            bg=COLOUR["panel"],
            fg=COLOUR["text_dim"],
        ).pack(anchor="w", padx=14, pady=(10, 2))

        self._output_var = tk.StringVar(value="")
        tk.Label(
            panel,
            textvariable=self._output_var,
            font=(FONT_FAMILY, 14),
            bg=COLOUR["panel"],
            fg=COLOUR["accent"],
            anchor="w",
            wraplength=620,
            justify=tk.LEFT,
        ).pack(fill=tk.X, padx=14, pady=(0, 6))

    # ── History panel ─────────────────────────────────────────────────────────

    def _build_history(self, parent):
        panel = tk.Frame(
            parent,
            bg=COLOUR["panel"],
            highlightbackground=COLOUR["border"],
            highlightthickness=1,
        )
        panel.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        header = tk.Frame(panel, bg=COLOUR["panel"])
        header.pack(fill=tk.X, padx=14, pady=(8, 4))

        tk.Label(
            header,
            text="PREDICTION HISTORY",
            font=(FONT_FAMILY, 9, "bold"),
            bg=COLOUR["panel"],
            fg=COLOUR["text_dim"],
        ).pack(side=tk.LEFT)

        _btn(header, "Export", self._on_export, width=7, small=True).pack(side=tk.RIGHT)

        self._history_box = scrolledtext.ScrolledText(
            panel,
            height=8,
            font=(FONT_FAMILY, 10),
            bg=COLOUR["bg"],
            fg=COLOUR["text"],
            insertbackground=COLOUR["text"],
            relief=tk.FLAT,
            padx=10,
            pady=6,
            state=tk.DISABLED,
        )
        self._history_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

    # ── Controls ──────────────────────────────────────────────────────────────

    def _build_controls(self, parent):
        panel = tk.Frame(
            parent,
            bg=COLOUR["panel"],
            highlightbackground=COLOUR["border"],
            highlightthickness=1,
        )
        panel.pack(fill=tk.X, ipady=10)

        inner = tk.Frame(panel, bg=COLOUR["panel"])
        inner.pack(padx=14, pady=(8, 2), fill=tk.X)

        # Start / Stop button
        self._start_btn_var = tk.StringVar(value="▶  Start")
        self._start_btn = _btn(
            inner,
            "",
            self._on_toggle,
            width=10,
            textvariable=self._start_btn_var,
            accent=True,
        )
        self._start_btn.pack(side=tk.LEFT, padx=(0, 16))

        # Confidence threshold slider
        tk.Label(
            inner,
            text="Confidence threshold:",
            font=(FONT_FAMILY, 10),
            bg=COLOUR["panel"],
            fg=COLOUR["text"],
        ).pack(side=tk.LEFT)

        self._threshold_var = tk.DoubleVar(value=0.85)
        self._threshold_label_var = tk.StringVar(value="0.85")

        slider = ttk.Scale(
            inner,
            from_=0.60,
            to=0.95,
            variable=self._threshold_var,
            orient=tk.HORIZONTAL,
            length=160,
            command=self._on_threshold_change,
        )
        slider.pack(side=tk.LEFT, padx=8)

        tk.Label(
            inner,
            textvariable=self._threshold_label_var,
            font=(FONT_FAMILY, 10, "bold"),
            bg=COLOUR["panel"],
            fg=COLOUR["accent"],
            width=4,
        ).pack(side=tk.LEFT)

    # ── Poll loop ─────────────────────────────────────────────────────────────

    def _poll(self):
        """
        Called every 50ms by Tkinter's scheduler.
        Drains the inference queue and updates UI.
        """
        if self._running:
            # Update connection status
            self._update_status()

            # Drain all pending predictions
            while True:
                result = self.engine.get_prediction()
                if result is None:
                    break
                self._handle_prediction(result)

        # Reschedule
        self.after(50, self._poll)

    def _handle_prediction(self, result: PredictionResult):
        """Process one PredictionResult from the engine."""

        # Update letter display
        colour = _confidence_colour(result.confidence)
        self._letter_var.set(result.letter)
        self._letter_label.configure(fg=colour)
        self._conf_val.set(result.confidence * 100)
        tag = " [J/Z]" if result.is_dynamic else ""
        self._conf_var.set(f"{result.confidence:.0%} confidence{tag}")

        # Log to history
        ts = datetime.fromtimestamp(result.timestamp).strftime("%H:%M:%S")
        entry = f"[{ts}]  {result.letter}   {result.confidence:.0%}"
        if result.is_dynamic:
            entry += "  ✦"
        self._append_history(entry)
        self._session_log.append(entry)

        # Append to word buffer (guard: only once per stable gesture)
        self._debounce_buf.append(result.letter)
        if len(self._debounce_buf) > 3:
            self._debounce_buf.pop(0)

        if (len(self._debounce_buf) == 3 and
                len(set(self._debounce_buf)) == 1 and
                result.letter != self._last_appended):
            self._word_buffer.append(result.letter)
            self._last_appended = result.letter
            self._word_var.set("".join(self._word_buffer))
        elif len(set(self._debounce_buf)) > 1:
            self._last_appended = ""

    def _update_status(self):
        status = self.engine.get_status()
        if status == "running":
            self._status_var.set("● Connected")
            self._status_label.configure(fg=COLOUR["green"])
        elif status == "no_hand":
            self._status_var.set("● No hand detected")
            self._status_label.configure(fg=COLOUR["amber"])
            self._letter_var.set("—")
            self._letter_label.configure(fg=COLOUR["grey"])
            self._conf_var.set("No gesture detected")
            self._conf_val.set(0)
        elif status == "error":
            self._status_var.set("● Error")
            self._status_label.configure(fg=COLOUR["red"])
        else:
            self._status_var.set("● Stopped")
            self._status_label.configure(fg=COLOUR["text_dim"])

    # ── Button callbacks ──────────────────────────────────────────────────────

    def _on_toggle(self):
        if not self._running:
            try:
                self.engine.start()
                self._running = True
                self._start_btn_var.set("■  Stop")
                self._status_var.set("● Connected")
                self._status_label.configure(fg=COLOUR["green"])
            except Exception as e:
                messagebox.showerror("Startup Error", str(e))
        else:
            self.engine.stop()
            self._running = False
            self._start_btn_var.set("▶  Start")
            self._letter_var.set("—")
            self._letter_label.configure(fg=COLOUR["grey"])
            self._conf_var.set("No gesture detected")
            self._conf_val.set(0)
            self._status_var.set("● Stopped")
            self._status_label.configure(fg=COLOUR["text_dim"])

    def _on_space(self):
        word = "".join(self._word_buffer).strip()
        if word:
            self._word_history.append(word)
            self._output_var.set(" ".join(self._word_history))
        self._word_buffer.clear()
        self._last_appended = ""
        self._debounce_buf.clear()
        self._word_var.set("")

    def _on_backspace(self):
        if self._word_buffer:
            self._word_buffer.pop()
            self._last_appended = self._word_buffer[-1] if self._word_buffer else ""
            self._word_var.set("".join(self._word_buffer))

    def _on_clear(self):
        self._word_buffer.clear()
        self._word_history.clear()
        self._debounce_buf.clear()
        self._last_appended = ""
        self._word_var.set("")
        self._output_var.set("")
        self._history_box.configure(state=tk.NORMAL)
        self._history_box.delete("1.0", tk.END)
        self._history_box.configure(state=tk.DISABLED)
        self._session_log.clear()

    def _on_threshold_change(self, _event=None):
        val = round(self._threshold_var.get(), 2)
        # Snap to nearest 0.05
        val = round(val / 0.05) * 0.05
        self._threshold_label_var.set(f"{val:.2f}")
        if self._running:
            self.engine.set_confidence_threshold(val)

    def _on_export(self):
        if not self._session_log:
            messagebox.showinfo("Export", "No predictions to export yet.")
            return
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        default  = f"asl_session_{ts}.txt"
        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialfile=default,
        )
        if not filepath:
            return
        with open(filepath, "w") as f:
            f.write(f"ASL Fingerspelling Session — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
            f.write("Output:\n")
            f.write(" ".join(self._word_history) + "\n\n")
            f.write("Prediction Log:\n")
            for entry in self._session_log:
                f.write(entry + "\n")
        messagebox.showinfo("Export", f"Session saved to:\n{filepath}")

    # ── History helpers ───────────────────────────────────────────────────────

    def _append_history(self, text: str):
        self._history_box.configure(state=tk.NORMAL)
        self._history_box.insert(tk.END, text + "\n")
        self._history_box.see(tk.END)
        self._history_box.configure(state=tk.DISABLED)

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def _on_close(self):
        if self._running:
            self.engine.stop()
        self.destroy()


# ── Helper widgets ────────────────────────────────────────────────────────────

def _btn(parent, text, command, width=8, small=False,
         textvariable=None, accent=False):
    """Styled button factory."""
    kwargs = dict(
        command=command,
        font=(FONT_FAMILY, 9 if small else 10, "bold"),
        bg=COLOUR["accent_dark"] if accent else COLOUR["border"],
        fg=COLOUR["white"],
        activebackground=COLOUR["accent"],
        activeforeground=COLOUR["white"],
        relief=tk.FLAT,
        cursor="hand2",
        padx=8,
        pady=4 if small else 6,
        width=width,
        bd=0,
    )
    if textvariable:
        kwargs["textvariable"] = textvariable
    else:
        kwargs["text"] = text
    return tk.Button(parent, **kwargs)


def _confidence_colour(confidence: float) -> str:
    """Return a display colour based on prediction confidence."""
    if confidence >= 0.85:
        return COLOUR["green"]
    elif confidence >= 0.70:
        return COLOUR["amber"]
    else:
        return COLOUR["text_dim"]


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = ASLApp()
    app.mainloop()