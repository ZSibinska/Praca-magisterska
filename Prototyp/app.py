import os
import sys
from typing import List, Dict, Optional

from PySide6.QtCore import Qt, QObject, Signal, QThread
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from chatbot_engine import (
    ADHDResearchEngine,
    MODEL_PATH,
    RAG_FILE_PATH,
    MODEL_DISPLAY_NAME,
    RAG_DISPLAY_NAME,
    MIN_CONVERSATION_WORDS,
    MAX_FOLLOWUPS,
)


def normalize_spaces(text: str) -> str:
    return " ".join(text.split())


def word_count(text: str) -> int:
    return len(normalize_spaces(text).split())


class ChatBubble(QFrame):
    def __init__(self, text: str, is_user: bool):
        super().__init__()
        self.setObjectName("bubble")

        outer_layout = QHBoxLayout(self)
        outer_layout.setContentsMargins(12, 4, 12, 4)

        bubble = QLabel(text)
        bubble.setWordWrap(True)
        bubble.setTextInteractionFlags(Qt.TextSelectableByMouse)
        bubble.setMaximumWidth(760)
        bubble.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)

        if is_user:
            bubble.setObjectName("userBubble")
            outer_layout.addStretch()
            outer_layout.addWidget(bubble)
        else:
            bubble.setObjectName("assistantBubble")
            outer_layout.addWidget(bubble)
            outer_layout.addStretch()


class Worker(QObject):
    finished = Signal(dict)
    error = Signal(str)

    def __init__(
        self,
        stage: str,
        user_text: str,
        symptoms_raw: str,
        conversation_turns: List[Dict[str, str]],
        followups_asked: int,
    ):
        super().__init__()
        self.stage = stage
        self.user_text = user_text
        self.symptoms_raw = symptoms_raw
        self.conversation_turns = conversation_turns
        self.followups_asked = followups_asked

    def run(self):
        try:
            engine = ADHDResearchEngine()

            if self.stage == "intake":
                assistant_question = "Jaki ciekawy film oglądałeś albo oglądałaś ostatnio?"
                updated_conversation_turns = [
                    {"role": "assistant", "text": assistant_question}
                ]

                result = {
                    "stage": "conversation",
                    "symptoms_raw": self.user_text,
                    "conversation_turns": updated_conversation_turns,
                    "followups_asked": self.followups_asked,
                    "assistant_messages": [assistant_question],
                    "analysis_done": False,
                    "user_summary": "",
                    "full_analysis": "",
                    "report_data": None,
                }
                self.finished.emit(result)
                return

            updated_conversation_turns = list(self.conversation_turns)
            updated_conversation_turns.append({"role": "user", "text": self.user_text})

            conversation_text = normalize_spaces(
                " ".join(
                    turn["text"]
                    for turn in updated_conversation_turns
                    if turn["role"] == "user"
                )
            )

            if (
                word_count(conversation_text) < MIN_CONVERSATION_WORDS
                and self.followups_asked < MAX_FOLLOWUPS
            ):
                followup = engine.generate_followup_question(
                    last_answer=self.user_text,
                    followups_asked=self.followups_asked,
                )

                updated_conversation_turns.append({"role": "assistant", "text": followup})

                result = {
                    "stage": "conversation",
                    "symptoms_raw": self.symptoms_raw,
                    "conversation_turns": updated_conversation_turns,
                    "followups_asked": self.followups_asked + 1,
                    "assistant_messages": [followup],
                    "analysis_done": False,
                    "user_summary": "",
                    "full_analysis": "",
                    "report_data": None,
                }
                self.finished.emit(result)
                return

            symptoms_summary = engine.summarize_symptoms(self.symptoms_raw)
            retrieved_context = engine.retrieve_context_for_analysis(
                self.symptoms_raw,
                conversation_text,
            )
            fragment_matches = engine.match_fragments_to_rag(
                conversation_text,
            )
            full_analysis = engine.generate_full_analysis(
                symptoms_raw=self.symptoms_raw,
                symptoms_summary=symptoms_summary,
                conversation_text=conversation_text,
                retrieved_context=retrieved_context,
                fragment_matches=fragment_matches,
            )
            user_summary = engine.generate_user_summary(full_analysis)
            final_assistant_message = (
                "Dziękuję. Analiza została zakończona, a podsumowanie umieszczono "
                "w dedykowanej sekcji poniżej czatu."
            )
            updated_conversation_turns.append({
                "role": "assistant",
                "text": final_assistant_message,
            })

            report_data = engine.create_report_payload(
                symptoms_raw=self.symptoms_raw,
                symptoms_summary=symptoms_summary,
                conversation_turns=updated_conversation_turns,
                retrieved_context=retrieved_context,
                fragment_matches=fragment_matches,
                full_analysis=full_analysis,
                user_summary=user_summary,
            )
            engine.save_report_files(report_data)

            result = {
                "stage": "done",
                "symptoms_raw": self.symptoms_raw,
                "conversation_turns": updated_conversation_turns,
                "followups_asked": self.followups_asked,
                "assistant_messages": [final_assistant_message],
                "analysis_done": True,
                "user_summary": user_summary,
                "full_analysis": full_analysis,
                "report_data": report_data,
            }
            self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chatbot badawczy")
        self.resize(1600, 950)
        self.showMaximized()

        self.stage = "intake"
        self.symptoms_raw = ""
        self.conversation_turns: List[Dict[str, str]] = []
        self.followups_asked = 0
        self.report_data: Optional[Dict] = None
        self.user_summary = ""

        self.thread: Optional[QThread] = None
        self.worker: Optional[Worker] = None

        self._build_ui()
        self._apply_styles()
        self._load_initial_messages()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # =========================
        # LEWY PANEL
        # =========================
        left_panel = QFrame()
        left_panel.setObjectName("panel")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(22, 22, 22, 22)
        left_layout.setSpacing(14)

        title = QLabel("Chatbot badawczy")
        title.setObjectName("title")

        subtitle = QLabel(
            "Prototyp umożliwia przeprowadzenie rozmowy badawczej oraz analizę cech językowych wypowiedzi użytkownika."
        )
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)

        role_info = QLabel(
            "<b>Rola badacza</b><br>"
            "Jako badacz możesz podyskutować z chatbotem z perspektywy pacjenta. "
            "W rozmowie należy więc symulować osobę zgłaszającą swoje trudności, obserwacje i doświadczenia."
        )
        role_info.setObjectName("infoCard")
        role_info.setWordWrap(True)

        model_info = QLabel(
            f"<b>Model lokalny</b><br>{MODEL_DISPLAY_NAME}"
        )
        model_info.setObjectName("infoCard")
        model_info.setWordWrap(True)
        model_info.setTextInteractionFlags(Qt.TextSelectableByMouse)

        rag_info = QLabel(
            f"<b>Baza wiedzy RAG</b><br>{RAG_DISPLAY_NAME}"
        )
        rag_info.setObjectName("infoCard")
        rag_info.setWordWrap(True)
        rag_info.setTextInteractionFlags(Qt.TextSelectableByMouse)

        report_info = QLabel(
            "Pełny raport analizy zapisywany jest automatycznie w tle. "
            "Krótkie podsumowanie pojawi się w sekcji „Wynik analizy”."
        )
        report_info.setObjectName("info")
        report_info.setWordWrap(True)

        self.new_chat_button = QPushButton("Nowa rozmowa")
        self.new_chat_button.clicked.connect(self.reset_chat)

        left_layout.addWidget(title)
        left_layout.addWidget(subtitle)
        left_layout.addSpacing(6)
        left_layout.addWidget(role_info)
        left_layout.addWidget(model_info)
        left_layout.addWidget(rag_info)
        left_layout.addSpacing(4)
        left_layout.addWidget(report_info)
        left_layout.addStretch()
        left_layout.addWidget(self.new_chat_button)

        # =========================
        # PRAWA CZĘŚĆ
        # =========================
        right_stack = QVBoxLayout()
        right_stack.setContentsMargins(0, 0, 0, 0)
        right_stack.setSpacing(10)

        # CHAT
        chat_panel = QFrame()
        chat_panel.setObjectName("panel")
        chat_layout_outer = QVBoxLayout(chat_panel)
        chat_layout_outer.setContentsMargins(0, 0, 0, 0)
        chat_layout_outer.setSpacing(0)

        self.chat_scroll = QScrollArea()
        self.chat_scroll.setWidgetResizable(True)
        self.chat_scroll.setFrameShape(QFrame.NoFrame)
        self.chat_scroll.setObjectName("chatScroll")

        self.chat_container = QWidget()
        self.chat_container.setObjectName("chatContainer")

        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.setContentsMargins(20, 20, 20, 20)
        self.chat_layout.setSpacing(10)
        self.chat_layout.addStretch()

        self.chat_scroll.setWidget(self.chat_container)

        input_wrap = QFrame()
        input_wrap.setObjectName("inputWrap")
        input_layout = QHBoxLayout(input_wrap)
        input_layout.setContentsMargins(20, 14, 20, 14)
        input_layout.setSpacing(12)

        self.input_line = QLineEdit()
        self.input_line.setPlaceholderText("Napisz odpowiedź...")
        self.input_line.returnPressed.connect(self.send_message)

        self.send_button = QPushButton("Wyślij")
        self.send_button.setObjectName("sendButton")
        self.send_button.clicked.connect(self.send_message)

        input_layout.addWidget(self.input_line, 1)
        input_layout.addWidget(self.send_button)

        chat_layout_outer.addWidget(self.chat_scroll, 1)
        chat_layout_outer.addWidget(input_wrap, 0)

        # WYNIK
        result_panel = QFrame()
        result_panel.setObjectName("panel")
        result_panel.setMaximumHeight(180)

        result_layout = QVBoxLayout(result_panel)
        result_layout.setContentsMargins(20, 18, 20, 18)
        result_layout.setSpacing(10)

        result_title = QLabel("Wynik analizy")
        result_title.setObjectName("sectionTitle")

        self.result_box = QLabel(
            "Po zakończeniu rozmowy tutaj pojawi się krótkie podsumowanie."
        )
        self.result_box.setObjectName("resultBox")
        self.result_box.setWordWrap(True)
        self.result_box.setTextInteractionFlags(Qt.TextSelectableByMouse)

        result_layout.addWidget(result_title)
        result_layout.addWidget(self.result_box)
        result_layout.addStretch()

        right_stack.addWidget(chat_panel, 9)
        right_stack.addWidget(result_panel, 1)

        right_wrap = QWidget()
        right_wrap.setLayout(right_stack)

        root.addWidget(left_panel, 1)
        root.addWidget(right_wrap, 5)

    def _apply_styles(self):
        self.setStyleSheet("""
            QMainWindow {
                background: #f3f4f6;
            }

            QFrame#panel {
                background: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 20px;
            }

            QScrollArea#chatScroll {
                border: none;
                background: transparent;
            }

            QWidget#chatContainer {
                background: #0b0f19;
                border-top-left-radius: 20px;
                border-top-right-radius: 20px;
            }

            QFrame#inputWrap {
                background: #ffffff;
                border-top: 1px solid #eceff3;
                border-bottom-left-radius: 20px;
                border-bottom-right-radius: 20px;
            }

            QLabel#title {
                font-size: 30px;
                font-weight: 700;
                color: #0f172a;
            }

            QLabel#subtitle {
                font-size: 15px;
                color: #64748b;
                line-height: 1.5;
            }

            QLabel#info {
                font-size: 14px;
                color: #334155;
                line-height: 1.6;
            }

            QLabel#infoCard {
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 14px;
                padding: 12px 14px;
                font-size: 14px;
                color: #334155;
                line-height: 1.5;
            }

            QLabel#sectionTitle {
                font-size: 18px;
                font-weight: 700;
                color: #0f172a;
            }

            QLabel#resultBox {
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 14px;
                padding: 14px;
                color: #0f172a;
                font-size: 14px;
            }

            QLabel#assistantBubble {
                background: #f8fafc;
                color: #0f172a;
                border-radius: 18px;
                padding: 14px 16px;
                font-size: 15px;
            }

            QLabel#userBubble {
                background: #0f172a;
                color: white;
                border-radius: 18px;
                padding: 14px 16px;
                font-size: 15px;
            }

            QLineEdit {
                border: 1px solid #cbd5e1;
                border-radius: 14px;
                padding: 12px 14px;
                background: #ffffff;
                font-size: 14px;
                color: #0f172a;
            }

            QPushButton {
                background: #0f172a;
                color: white;
                border: none;
                border-radius: 14px;
                padding: 11px 16px;
                font-weight: 600;
            }

            QPushButton:hover:!disabled {
                background: #1e293b;
            }

            QPushButton:disabled {
                background: #94a3b8;
                color: #e2e8f0;
            }

            QPushButton#sendButton {
                min-width: 92px;
            }
        """)

    def _load_initial_messages(self):
        self.add_assistant_message(
            "Cześć! Witaj w badawczej wersji chatbota do analizy cech językowych."
        )
        self.add_assistant_message(
            "Na początku zapytam Cię o Twoją perspektywę, a potem przejdziemy do swobodnej rozmowy."
        )
        self.add_assistant_message(
            "Jakie objawy u siebie zauważasz i dlaczego chcesz się diagnozować?"
        )

    def add_bubble(self, text: str, is_user: bool):
        bubble = ChatBubble(text, is_user=is_user)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, bubble)
        self.scroll_chat_to_bottom()

    def add_user_message(self, text: str):
        self.add_bubble(text, True)

    def add_assistant_message(self, text: str):
        self.add_bubble(text, False)

    def scroll_chat_to_bottom(self):
        QApplication.processEvents()
        bar = self.chat_scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

    def set_ui_busy(self, busy: bool):
        self.input_line.setEnabled(not busy)
        self.send_button.setEnabled(not busy)
        self.new_chat_button.setEnabled(not busy)

        if busy:
            self.input_line.setPlaceholderText("Przetwarzanie...")
        else:
            self.input_line.setPlaceholderText("Napisz odpowiedź...")

    def send_message(self):
        user_text = self.input_line.text().strip()
        if not user_text:
            return

        self.input_line.clear()
        self.add_user_message(user_text)
        self.set_ui_busy(True)

        self.thread = QThread()
        self.worker = Worker(
            stage=self.stage,
            user_text=user_text,
            symptoms_raw=self.symptoms_raw,
            conversation_turns=self.conversation_turns,
            followups_asked=self.followups_asked,
        )
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.error.connect(self.on_worker_error)

        self.worker.finished.connect(self.thread.quit)
        self.worker.error.connect(self.thread.quit)

        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def on_worker_finished(self, result: Dict):
        self.stage = result["stage"]
        self.symptoms_raw = result["symptoms_raw"]
        self.conversation_turns = result["conversation_turns"]
        self.followups_asked = result["followups_asked"]

        for message in result["assistant_messages"]:
            self.add_assistant_message(message)

        if result["analysis_done"]:
            self.user_summary = result["user_summary"]
            self.report_data = result["report_data"]
            self.result_box.setText(self.user_summary)

        self.set_ui_busy(False)

        if self.stage == "done":
            self.input_line.setEnabled(False)
            self.send_button.setEnabled(False)
            self.input_line.setPlaceholderText(
                "Rozmowa zakończona. Wybierz „Nowa rozmowa”, aby rozpocząć od nowa."
            )

    def on_worker_error(self, error_text: str):
        self.set_ui_busy(False)
        QMessageBox.critical(self, "Błąd", error_text)

    def reset_chat(self):
        self.stage = "intake"
        self.symptoms_raw = ""
        self.conversation_turns = []
        self.followups_asked = 0
        self.report_data = None
        self.user_summary = ""

        while self.chat_layout.count() > 1:
            item = self.chat_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self.result_box.setText(
            "Po zakończeniu rozmowy tutaj pojawi się krótkie podsumowanie."
        )
        self.input_line.setEnabled(True)
        self.send_button.setEnabled(True)
        self.new_chat_button.setEnabled(True)
        self.input_line.setPlaceholderText("Napisz odpowiedź...")

        self._load_initial_messages()

    def closeEvent(self, event):
        if self.thread is not None and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()
        event.accept()


def check_requirements() -> Optional[str]:
    if not os.path.exists(MODEL_PATH):
        return f"Nie znaleziono modelu GGUF pod ścieżką: {MODEL_PATH}"
    if not os.path.exists(RAG_FILE_PATH):
        return f"Nie znaleziono pliku wiedzy RAG: {RAG_FILE_PATH}"
    return None


def main():
    error = check_requirements()

    app = QApplication(sys.argv)
    app.setApplicationName("Chatbot badawczy")
    app.setFont(QFont("Segoe UI", 10))

    if error:
        QMessageBox.critical(None, "Błąd konfiguracji", error)
        sys.exit(1)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()