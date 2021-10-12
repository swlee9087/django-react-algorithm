package shop.cofin.api.api.item.domain;


import com.sun.istack.NotNull;
import lombok.Data;
import org.springframework.stereotype.Component;
import shop.cofin.api.api.board.domain.Article;

import javax.persistence.*;
import java.util.ArrayList;
import java.util.List;

@Data
@Component
@Entity
@Table(name="items")
public class Item {

    @Id
    @Column(name="item_id")
    @GeneratedValue
    private long itemId;
    @Column(name="item_brand") @NotNull    private String itemBrand;
    @Column(name="item_name") @NotNull    private String itemName;
    @Column(name="item_color") @NotNull    private String itemColor;
    @Column(name="release_date") @NotNull    private String releaseDate;
    @OneToMany(mappedBy = "item") private List<Article> articleList = new ArrayList<>();
}
