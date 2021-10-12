package shop.cofin.api.api.board.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import shop.cofin.api.api.board.domain.Article;

@Repository
public interface BoardRepository extends JpaRepository<Article, Long> {
}
